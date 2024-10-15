import subprocess
from watchgod import watch

def run_daphne():
    # Command to run daphne server
    cmd = ['daphne', '-b', '0.0.0.0', '-p', '8000', 'streamapp.asgi:application']
    process = subprocess.Popen(cmd)
    return process

if __name__ == "__main__":
    process = run_daphne()
    try:
        for changes in watch('streamapp'):
            print("Changes detected, restarting...")
            process.terminate()
            process.wait()
            process = run_daphne()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
