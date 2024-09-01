from . import BaseController
import numpy as np

class Controller(BaseController):

    def __init__(self):
        # PID gains
        self.p = 0.00100
        self.i = 0.132
        self.d = 5.00

        # Feedforward gain
        self.k_ff = 2.500

        # Model Predictive Control parameters
        self.horizon = 5
        self.Q_diag = [9.2854, 1.6106]
        self.R = 0.0100

        # Kalman Filter parameters
        self.Q_kf_diag = [20.0000, 0.0]
        self.R_kf =  0.0100

        # Adaptive parameters
        self.min_p = 0.3919
        self.max_p = 1.8326
        self.gain_factor = 0.286
        
        # Output smoothing
        self.alpha_out = 0.03105

        # Anti-windup
        self.max_integral = 3.8069

        self.smoothing_factor = 0.965

        # Error derivative smoothing
        self.alpha_derivative = 0.0152
        self.prev_d_term = 0

        # Kalman Filter state
        self.A = np.array([[1, 0.1], [0, 1]])
        self.C = np.array([[1, 0]])
        self.P = np.eye(2)
        self.x = np.zeros(2)
        self.error_integral = 0
        self.prev_error = 0
        self.prev_output = 0

        # Adaptive control parameters
        self.adaptive_gain = 0.01
        self.reference_model_pole = -20
        self.theta = np.zeros(3)  # Adaptive parameters [a, b, c]
        self.gamma = np.diag([0.10, 0.1, 0.1])  # Adaptation rate matrix
        
        self.lambda_smc = 0.749
        self.eta_smc = 0.06809
        self.boundary_layer = 0.0088


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

        # Adaptive control
        adaptive_term = self.adaptive_control(target_lataccel, current_lataccel)
        smc_term = self.sliding_mode_control(target_lataccel, current_lataccel)


        # Combine all terms
        output = p_term + i_term + d_term + ff_term + mpc_term  + smc_term + adaptive_term

        # Smooth output
        smoothed_output = (self.alpha_out * output) + ((1 - self.alpha_out) * self.prev_output)

        smoothed_output = np.clip(smoothed_output, -2, 2)
        # Update previous values
        self.prev_error = error
        self.prev_d_term = d_raw
        self.prev_output = smoothed_output

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
    
    def sliding_mode_control(self, target_lataccel, current_lataccel):
        s = self.lambda_smc * (target_lataccel - current_lataccel) + (self.x[1] - 0)  # Sliding surface
        sat_s = np.clip(s / self.boundary_layer, -1, 1)  # Saturation function
        u_smc = self.eta_smc * sat_s
        return u_smc

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

        for u in np.linspace(-1, 1, 100):
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

    def adaptive_control(self, target_lataccel, current_lataccel):
        # Calculate reference model output
        y_m = self.reference_model(target_lataccel)

        # Calculate tracking error
        e = current_lataccel - y_m

        # Update adaptive parameters
        phi = np.array([current_lataccel, target_lataccel, 1])
        self.theta += self.gamma @ phi * e * self.adaptive_gain

        # Calculate adaptive control input
        u_ad = np.dot(self.theta, phi)

        return u_ad

    def reference_model(self, target_lataccel):
        # Simple first-order reference model
        y_m = target_lataccel * (1 - np.exp(self.reference_model_pole * 0.1))
        return y_m