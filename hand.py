MUJOCO_XML_PATH = "shadow_hand/scene_right.xml" 
import mujoco
import importlib
importlib.reload(mujoco)
import mujoco.viewer
import numpy as np
import time
from main import EMGInferenceEngine  # Ensure this file contains your class or copy the class here

# === USER CONFIGURATION ===
MODEL_PATH = "rnn_emg_model_enhanced.pth"  # 'rnn_emg_model_enhanced' or 'transformer_emg_model_enhanced.pth'
SCALER_EMG_PATH = "scaler_emg.pkl"
SCALER_GLOVE_PATH = "scaler_glove.pkl"
MODEL_TYPE = "rnn"  # or "rnn" or "transformer"
SEQ_LENGTH = 80
N_CHANNELS = 10

# === INITIALIZE INFERENCE ENGINE ===
inference_engine = EMGInferenceEngine(
    model_path=MODEL_PATH,
    scaler_emg_path=SCALER_EMG_PATH,
    scaler_glove_path=SCALER_GLOVE_PATH,
    model_type=MODEL_TYPE
)

# === LOAD MuJoCo MODEL ===
print("Reloading model...")
model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
data = mujoco.MjData(model)
print(f"Model reloaded. Has {model.nu} actuators.")

# Check actuator setup
print(f"Model has {model.nu} actuators (expected: >=22 for full glove mapping).")

# === INFERENCE AND SIMULATION LOOP ===
def simulate_predicted_hand_movement():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Running MuJoCo ShadowHand simulation with EMG-based predictions...")

        while viewer.is_running():
            # Replace with real EMG input or stream as needed
            emg_sequence = np.random.randn(SEQ_LENGTH, N_CHANNELS)

            # Predict joint positions
            glove_prediction = inference_engine.predict(emg_sequence)

            # Safety clip to actuator control limits
            ctrl_len = min(len(glove_prediction), model.nu)
            glove_prediction = np.clip(
                glove_prediction[:ctrl_len],
                model.actuator_ctrlrange[:ctrl_len, 0],
                model.actuator_ctrlrange[:ctrl_len, 1]
            )

            # Apply control signal
            data.ctrl[:ctrl_len] = glove_prediction

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(1.0 / 60.0)  # ~60 FPS

# === RUN ===
if __name__ == "__main__":
    print("Reloading model from XML...")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)
    print("Model reloaded!")
    simulate_predicted_hand_movement()
