import subprocess
import os

def run_all_baselines():
    print("=== Baselines Calistiriliyor ===")
    
    print("\n[1/3] Vanilla LSTM egitiliyor...")
    subprocess.run(["python", "vanilla_lstm.py"], cwd="baselines")
    
    print("\n[2/3] GRU egitiliyor...")
    subprocess.run(["python", "gru_model.py"], cwd="baselines")
    
    print("\n[3/3] Pure ODE hesaplaniyor...")
    subprocess.run(["python", "pure_asm2d_baseline.py"], cwd="baselines")
    
    print("\n=== Tum Baselines Tamamlandi ===")

if __name__ == "__main__":
    if not os.path.exists("baselines"):
        print("Hata: 'baselines' klasoru bulunamadi!")
    else:
        run_all_baselines()
