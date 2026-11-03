"""Continue the authorized fixed sequence after Task A completes; never publishes or deploys."""
import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]

def run(name, script, *arguments):
    completion = OUT/f'{name}_pipeline_done.json'
    if completion.exists():
        print('Already completed:',name,flush=True)
        return
    state = {'stage':name,'status':'RUNNING','time':time.time(),'command':[sys.executable,str(OUT/'scripts'/script),*arguments]}
    (OUT/'ACTIVE_SAVEPOINT.json').write_text(json.dumps(state,indent=2)+'\n')
    print('START',name,flush=True)
    started = time.time()
    log = OUT/f'{name}.log'
    assert not log.exists(), f'Refusing to replace log: {log}'
    with log.open('w',encoding='utf-8') as stream:
        result = subprocess.run(state['command'],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    state.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,elapsed_seconds=time.time()-started)
    (OUT/'ACTIVE_SAVEPOINT.json').write_text(json.dumps(state,indent=2)+'\n')
    assert result.returncode==0, f'{name} failed with exit {result.returncode}; inspect {log}'
    completion.write_text(json.dumps(state,indent=2)+'\n')
    print('COMPLETE',name,flush=True)

def main():
    assert (OUT/'task_a_completion.json').exists(), 'Task A must finish before downstream training'
    # The first runner revision recorded model.overrides after train(), which points at its final best.
    # Keep that value separately and record the actual resume input proven by resume_verified.json.
    completion_path = OUT/'task_a_completion.json'
    completion = json.loads(completion_path.read_text())
    expected_start = str(OUT/'preserved_inputs/morning_original_last.pt')
    if completion['start_model'] != expected_start:
        completion['model_after_training'] = completion['start_model']
        completion['start_model'] = expected_start
        completion['starting_model_evidence'] = 'resume_verified.json and task_a/args.yaml; first receipt recorded final model path'
        completion_path.write_text(json.dumps(completion,indent=2)+'\n')
    run('task_a_evaluation','evaluate_task_a.py')
    selection = json.loads((OUT/'task_a_comparison.json').read_text())
    run('field_condition_training','train_task.py','field_condition','--condition-start',selection['selected_path'])
    run('field_harvest_training','train_task.py','field_harvest')
    run('field_prop_training','train_task.py','field_prop')
    run('field_evaluation','evaluate_field.py')
    (OUT/'ACTIVE_SAVEPOINT.json').write_text(json.dumps({'stage':'TRAINING_AND_EVALUATION_COMPLETE','status':'AWAITING_FINAL_VALIDATION_AND_PUBLICATION','time':time.time()},indent=2)+'\n')

if __name__ == '__main__':
    main()
