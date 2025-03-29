import joblib

checkpoint_path = 'study_checkpoint.pkl'
try:
    study = joblib.load(checkpoint_path)
    print(f"Checkpoint loaded successfully. Type: {type(study)}")
    print(study)
except Exception as e:
    print(f"Error loading checkpoint: {e}")