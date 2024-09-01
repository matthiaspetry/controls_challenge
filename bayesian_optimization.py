import argparse
import importlib
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile

from io import BytesIO
from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from controllers import BaseController

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args




sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent / "data"

class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value: Union[float, np.ndarray, List[float]]) -> Union[int, np.ndarray]:
        value = self.clip(value)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.bins[token]

    def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
        return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])

class TinyPhysicsModel:
    def __init__(self, model_path: str, debug: bool) -> None:
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = 'CPUExecutionProvider'

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample

    def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
        tokenized_actions = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.8))

class TinyPhysicsSimulator:
    def __init__(self, model: TinyPhysicsModel, data_path: str, controller: BaseController, debug: bool = False) -> None:
        self.data_path = data_path
        self.sim_model = model
        self.data = self.get_data(data_path)
        self.controller = controller
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[-1]
        seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
        np.random.seed(seed)

    def get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values
        })
        return processed_df

    def sim_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def control_step(self, step_idx: int) -> None:
        action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], future_plan=self.futureplan)
        if step_idx < CONTROL_START_IDX:
            action = self.data['steer_command'].values[step_idx]
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

    def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
            state['target_lataccel'],
            FuturePlan(
                lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
            )
        )

    def step(self) -> None:
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan
        self.control_step(self.step_idx)
        self.sim_step(self.step_idx)
        self.step_idx += 1

    def plot_data(self, ax, lines, axis_labels, title) -> None:
        ax.clear()
        for line, label in lines:
            ax.plot(line, label=label)
        ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
        ax.legend()
        ax.set_title(f"{title} | Step: {self.step_idx}")
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    def compute_cost(self) -> Dict[str, float]:
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

    def rollout(self) -> Dict[str, float]:
        if self.debug:
            plt.ion()
            fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

        for _ in range(CONTEXT_LENGTH, len(self.data)):
            self.step()
            if self.debug and self.step_idx % 10 == 0:
                print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
                self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
                self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
                self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
                self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
                plt.pause(0.01)

        if self.debug:
            plt.ioff()
            plt.show()
        return self.compute_cost()

def get_available_controllers():
    return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

def run_rollout(data_path, controller, model_path, debug=False):
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
    sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug)
    return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history

def download_dataset():
    print("Downloading dataset (0.6G)...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as z:
            for member in z.namelist():
                if not member.endswith('/'):
                    with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:
                        dest.write(src.read())


class AdvancedController(BaseController):
    def __init__(self, params):
        # PID gains
        self.p = params['p']
        self.i = params['i']
        self.d = params['d']
        
        # Feedforward gain
        self.k_ff = params['k_ff']
        
        # Model Predictive Control parameters
        self.horizon = int(5)
        self.Q_diag = [params['Q_1'], params['Q_2']]
        self.R = params['R']
        
        # Kalman Filter parameters
        self.Q_kf_diag = [params['Q_kf_1'], params['Q_kf_2']]
        self.R_kf = params['R_kf']
        
        # Adaptive parameters
        self.min_p = params['min_p']
        self.max_p = params['max_p']
        self.gain_factor = params['gain_factor']
        
        # Output smoothing
        self.alpha_out = params['alpha_out']
        self.jerk_weight = params["jerk_weight"]
        
        # Anti-windup
        self.max_integral = params['max_integral']
        
        self.gain_factor_i = params['gain_factor_i']
        self.min_i = params['min_i']
        self.max_i = params['max_i']
        self.alpha_derivative = params['alpha_derivative']
        
        # Initialize other necessary attributes
        self.A = np.array([[1, 0.1], [0, 1]])
        self.C = np.array([[1, 0]])
        self.P = np.eye(2)
        self.x = np.zeros(2)
        self.error_integral = 0
        self.prev_error = 0
        self.prev_output = 0
        self.prev_d_term = 0
        self.prev_output_rate =0 

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Kalman Filter update
        z = np.array([current_lataccel])
        self.x, self.P = self.kalman_filter(z)

        # Calculate error
        error = target_lataccel - self.x[0]

        # Adapt gains
        self.adapt_gains(error)

        # PID control with derivative smoothing
        p_term = self.p * error
        self.error_integral = np.clip(self.error_integral + error, -self.max_integral, self.max_integral)
        i_term = self.i * self.error_integral
        d_raw = error - self.prev_error
        d_term = self.d * ((1 - self.alpha_derivative) * d_raw + self.alpha_derivative * self.prev_d_term)

        # Feedforward
        ff_term = self.calculate_feedforward(target_lataccel, future_plan)

        # Model Predictive Control
        mpc_term = self.model_predictive_control(target_lataccel, future_plan)

        # Jerk minimization
        output_rate = (p_term + i_term + d_term + ff_term) - self.prev_output
        jerk_term = self.jerk_weight * (output_rate - self.prev_output_rate)

        # Combine all terms
        output = p_term + i_term + d_term + ff_term + mpc_term - jerk_term

        # Smooth output
        smoothed_output = (self.alpha_out * output) + ((1 - self.alpha_out) * self.prev_output)

        # Update previous values
        self.prev_error = error
        self.prev_d_term = d_raw
        self.prev_output = smoothed_output
        self.prev_output_rate = output_rate


        return smoothed_output

    def kalman_filter(self, z):
        Q_kf = np.diag(self.Q_kf_diag)
        # Prediction
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + Q_kf

        # Update
        y = z - self.C @ x_pred
        S = self.C @ P_pred @ self.C.T + self.R_kf
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        x_updated = x_pred + K @ y
        P_updated = (np.eye(2) - K @ self.C) @ P_pred

        return x_updated, P_updated

    def model_predictive_control(self, target_lataccel, future_plan):
        if future_plan is None or len(future_plan.lataccel) == 0:
            return 0

        future_targets = np.array(future_plan.lataccel)
        if len(future_targets) < self.horizon:
            future_targets = np.pad(future_targets, (0, self.horizon - len(future_targets)), 'edge')
        else:
            future_targets = future_targets[:self.horizon]

        current_state = self.x

        Q = np.diag(self.Q_diag)
        best_control = 0
        min_cost = float('inf')

        for u in np.linspace(-1, 1, 20):
            cost = 0
            x = current_state
            for i in range(self.horizon):
                error = np.array([future_targets[i] - x[0], 0])
                cost += error.T @ Q @ error + self.R * u**2
                x = self.A @ x + np.array([0, u])

            if cost < min_cost:
                min_cost = cost
                best_control = u

        return best_control

    def calculate_feedforward(self, current_target, future_plan):
        if future_plan is None or len(future_plan.lataccel) == 0:
            return 0
        future_window = min(5, len(future_plan.lataccel))
        future_change = np.mean(future_plan.lataccel[:future_window]) - current_target
        return self.k_ff * future_change

    def adapt_gains(self, error):
        error_magnitude = abs(error)
        self.p = np.clip(self.p + error_magnitude * self.gain_factor, self.min_p, self.max_p)
        self.max_integral = np.clip(self.max_integral * (1 + error_magnitude * 0.01), 1, 10)




import os
from datetime import datetime

def write_params_to_file(params, total_cost):
    filename = "best_params_log2.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a') as f:
        f.write(f"\n--- New Best Parameters Found at {timestamp} ---\n")
        f.write(f"Average Cost: {total_cost:.4f}\n\n")
        for param, value in params.items():
            f.write(f"{param}: {value:.4f}\n")
        f.write("-" * 50 + "\n")
    
    print(f"Parameters appended to {filename}")


def bayesian_optimization(num_iterations, run_rollout_func, data_paths, model_path):
    # Define the search space
    space = [
        Real(0.1, 5.0, name='p'),
        Real(0.01, 5.0, name='i'),
        Real(0.1, 5.0, name='d'),
        Real(0.1, 2.0, name='k_ff'),
        Real(0.1, 10.0, name='Q_1'),
        Real(0.01, 10.0, name='Q_2'),
        Real(0.01, 1.0, name='R'),
        Real(0.01, 10.0, name='Q_kf_1'),
        Real(0.01, 10.0, name='Q_kf_2'),
        Real(0.01, 10.0, name='R_kf'),
        Real(0.1, 3.0, name='min_p'),
        Real(1.0, 5.0, name='max_p'),
        Real(0.1, 2.0, name='min_i'),
        Real(1.0, 5.0, name='max_i'),
        Real(0.001, 1.0, name='jerk_weight'),
        Real(0.1, 2.0, name='gain_factor'),
        Real(0.1, 2.0, name='gain_factor_i'),
        Real(0.01, 1.0, name='alpha_derivative'),
        Real(0.01, 1.0, name='alpha_out'),
        Real(1.0, 10.0, name='max_integral')
   
    ]

    # Define the objective function
    @use_named_args(space)
    def objective(**params):
        # Run the simulation with the custom controller over multiple files
        controller = AdvancedController(params)
        run_rollout_partial = partial(run_rollout_func, controller=controller, model_path=model_path, debug=False)
        results = process_map(run_rollout_partial, data_paths, max_workers=16, chunksize=16)
        total_cost = sum([result[0]['total_cost'] for result in results])
        
        avg_cost = total_cost / len(data_paths)
        
        if avg_cost < 80:
            write_params_to_file(params, avg_cost)

        # Return the average total cost for the optimizer to minimize
        return avg_cost
    

    # Run Bayesian optimization
    res = gp_minimize(objective, space, n_calls=num_iterations, random_state=420, verbose=True)

    # Extract the best parameters and the corresponding cost
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    best_cost = res.fun

    return best_params, best_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for random search")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        download_dataset()

    data_path = Path(args.data_path)

    if data_path.is_file():
        
        print(f"Performing random search optimization with {args.num_iterations} iterations...")
        best_params, best_cost = bayesian_optimization(args.num_iterations, run_rollout, [data_path], args.model_path)

        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value:.4f}")
        print(f"Best total cost: {best_cost:.4f}")

    elif data_path.is_dir():
        data_files = sorted(data_path.iterdir())[:args.num_segs]

        print(f"Performing random search optimization with {args.num_iterations} iterations over {len(data_files)} files...")
        best_params, best_cost = bayesian_optimization(args.num_iterations, run_rollout, data_files, args.model_path)

        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value:.4f}")
        print(f"Best total cost: {best_cost:.4f}")
        data_files = sorted(data_path.iterdir())[:5000]
        controller = CustomController(best_params)
        run_rollout_partial = partial(run_rollout, controller=controller, model_path=args.model_path, debug=False)
        results = process_map(run_rollout_partial, data_files, max_workers=16, chunksize=16)
        total_cost = sum([result[0]['total_cost'] for result in results])
        cost = total_cost / len(data_files)

        # Return the average total cost for the optimizer to minimize
        print(f"Total cost for 5000 rollouts: {cost:.4f}")
