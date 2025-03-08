class PIDController:

    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits

        # Initialize terms
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_error = 0.0
        self.previous_time = None


    def compute(self, error, current_time):

        # Calculate time difference
        if self.previous_time is None:
            dt = 0.0
        else:
            dt = current_time - self.previous_time

        # Update integral and derivative terms
        derivative = self.filtered_error - self.previous_error
        self.integral += error * dt

        self.filtered_error = 0.2 * self.previous_error + 0.8 * error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)

        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # Update previous values
        self.previous_error = error
        self.previous_time = current_time

        return output