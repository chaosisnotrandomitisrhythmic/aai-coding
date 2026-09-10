import json

import pytest
from shutil import which

from aai_coding.harness import bash_guard_msg, claude_air, claude_drop_sentinel, claude_slop, synthetic_resume


def test_bash_guard():
    assert bash_guard_msg('pytest -q | tail -5')
    assert bash_guard_msg('foo|head -n 3')
    assert bash_guard_msg('foo | tail -19')
    assert bash_guard_msg('foo | tail -n19')
    assert bash_guard_msg('pytest -q | tail -20') is None
    assert bash_guard_msg('pytest -q | tail -50') is None
    assert bash_guard_msg('head -3 file.txt') is None      # no pipe: a file slice, not output truncation
    assert bash_guard_msg('ls') is None
    # bare head/tail default to 10 lines, so they truncate exactly as hard as `-10`
    assert bash_guard_msg('ls ~ | head')
    assert bash_guard_msg('ls ~ | tail')
    assert bash_guard_msg('cat big.log | head | wc -l')
    assert bash_guard_msg('ls | head; echo done')
    assert bash_guard_msg('ls | head && echo ok')
    assert bash_guard_msg('ls | head\necho next')
    assert bash_guard_msg('ls | header') is None           # a command that merely starts with `head`
    assert bash_guard_msg('tail -f app.log') is None       # no pipe, and follow mode truncates nothing
    assert bash_guard_msg('cat f | head -c 200') is None   # bytes, not lines: out of scope either way
    assert bash_guard_msg('pytest -q >meta/stdout.txt 2>meta/stderr.txt') is None
    # fork default: the stderr-merge rule is off, so these are the caller's business
    assert bash_guard_msg('pytest -q 2>&1 | tail -20') is None
    assert bash_guard_msg('maturin develop 2>&1') is None
    assert bash_guard_msg('foo 2> &1') is None
    assert bash_guard_msg('make >> "$LOG" 2>&1') is None
    assert bash_guard_msg('ls -d a b c 2>&1') is None
    assert bash_guard_msg('pytest -q 2>&1 | tail -5')   # truncation still bites through a merge


def test_bash_guard_stderr_optin(monkeypatch):
    "AAI_BASH_GUARD_STDERR=1 restores the upstream stderr-merge rejection"
    monkeypatch.setenv('AAI_BASH_GUARD_STDERR', '1')
    assert bash_guard_msg('maturin develop 2>&1')
    assert bash_guard_msg('foo 2> &1')
    assert bash_guard_msg('pytest -q 2>&1 | tail -20')
    assert bash_guard_msg('pytest -q >meta/stdout.txt 2>meta/stderr.txt') is None




def test_synthetic_resume(tmp_path):
    t = tmp_path/'t.jsonl'
    boundary = json.dumps(dict(type='system', subtype='compact_boundary'))
    start = json.dumps(dict(type='attachment', attachment=dict(hookEvent='SessionStart')))
    t.write_text(f'{start}\nnot json\n{boundary}\n')
    assert synthetic_resume(t)
    t.write_text(f'{boundary}\n{start}\n')
    assert not synthetic_resume(t)
    assert not synthetic_resume(tmp_path/'missing.jsonl')


def test_claude_air(tmp_path, monkeypatch, capsys):
    "Nudge at 8 tool rounds, renudge every 5, reset on substantive text (>=100 chars across a message's deltas) or a new prompt"
    monkeypatch.setenv('LLMDOJO_STATE_DIR', str(tmp_path))
    def batch(): claude_air(dict(hook_event_name='PostToolBatch', session_id='s1', tool_calls=[]))
    def out(): return capsys.readouterr().out
    for _ in range(7): batch()
    assert out() == ''                                      # under threshold: silent
    batch()
    assert 're-points your attention' in json.loads(out())['hookSpecificOutput']['additionalContext']
    for _ in range(4): batch()
    assert out() == ''                                      # renudge interval not yet reached
    batch()
    assert 'made 13 tool rounds' in json.loads(out())['hookSpecificOutput']['additionalContext']
    claude_air(dict(hook_event_name='MessageDisplay', session_id='s1', message_id='m1', final=False, delta='x'*60))
    claude_air(dict(hook_event_name='MessageDisplay', session_id='s1', message_id='m1', final=True, delta='x'*60))
    batch()
    assert out() == ''                                      # accumulated 120 chars reset the counter
    claude_air(dict(hook_event_name='MessageDisplay', session_id='s1', message_id='m2', final=True, delta='short'))
    for _ in range(6): batch()
    assert out() == ''                                      # a sub-100-char message resets nothing: 7 rounds now
    batch()
    assert 'made 8 tool rounds' in json.loads(out())['hookSpecificOutput']['additionalContext']
    claude_air(dict(hook_event_name='UserPromptSubmit', session_id='s1', prompt='hi'))
    batch()
    assert out() == ''                                      # new prompt reset


def _transcript(path, blocks_per_msg, prompt_uuid='u1'):
    "Write a transcript: one user prompt, then one assistant record per list of block types"
    recs = [dict(type='user', uuid=prompt_uuid, message=dict(content='do the thing'))]
    recs += [dict(type='assistant', message=dict(content=[dict(type=t) for t in bs])) for bs in blocks_per_msg]
    path.write_text('\n'.join(json.dumps(r) for r in recs))


def test_drop_sentinel(tmp_path, monkeypatch, capsys):
    "Report new scars once across batch and stop events, resetting the count for each turn"
    monkeypatch.setenv('LLMDOJO_STATE_DIR', str(tmp_path))
    tp = tmp_path/'t.jsonl'
    def fire(ev='PostToolBatch', **kw): claude_drop_sentinel(dict(hook_event_name=ev, session_id='s1', transcript_path=str(tp), **kw))
    def out(): return capsys.readouterr().out
    _transcript(tp, [['thinking', 'text', 'tool_use'], ['thinking', 'tool_use']])
    fire()
    assert out() == ''                                      # clean turn: no adjacent thinking
    _transcript(tp, [['thinking', 'thinking', 'tool_use']])
    fire()
    r = json.loads(out())['hookSpecificOutput']
    assert r['hookEventName'] == 'PostToolBatch' and 'thinking blocks in a row' in r['additionalContext'] and 'end the turn' in r['additionalContext']
    fire()
    assert out() == ''                                      # already claimed: silent
    fire('Stop')
    assert out() == ''                                      # backstop silent after the batch boundary reported
    _transcript(tp, [['thinking', 'thinking', 'tool_use'], ['thinking', 'thinking', 'tool_use']])
    fire()
    assert 'thinking blocks in a row' in json.loads(out())['hookSpecificOutput']['additionalContext']   # only the fresh hole
    _transcript(tp, [['thinking', 'thinking', 'tool_use']]*3)
    fire('Stop')
    r = json.loads(out())
    assert r['decision'] == 'block' and 'thinking blocks in a row' in r['reason'] and 'state that missing thing now' in r['reason']
    fire('Stop')
    assert out() == ''                                      # blocks once, not forever
    _transcript(tp, [['thinking', 'thinking', 'tool_use']], prompt_uuid='u2')
    fire()
    assert 'thinking blocks in a row' in json.loads(out())['hookSpecificOutput']['additionalContext']   # new turn: count restarts


@pytest.mark.skipif(not which('slopometer'), reason='slopometer not installed')
def test_slop(tmp_path, monkeypatch, capsys):
    "Buffer message deltas, score only the final message, and report it once"
    monkeypatch.setenv('LLMDOJO_STATE_DIR', str(tmp_path))
    def disp(mid, txt, final=True, **kw): claude_slop(dict(hook_event_name='MessageDisplay', session_id='s1', message_id=mid, delta=txt, final=final, **kw))
    def psub(**kw): claude_slop(dict(hook_event_name='UserPromptSubmit', session_id='s1', **kw))
    def out(): return capsys.readouterr().out
    sloppy = "This isn't just a linter - it's a comprehensive paradigm that will streamline your workflow. " * 3
    disp('m1', 'a mid-turn note that nobody should score')
    disp('m2', sloppy[:40], final=False)
    disp('m2', sloppy[40:])
    assert out() == ''                                      # display tracking is silent
    psub()
    r = json.loads(out())['hookSpecificOutput']
    assert 'previous turn' in r['additionalContext'] and 'splice' in r['additionalContext']
    psub()
    assert out() == ''                                      # the same message reports once
