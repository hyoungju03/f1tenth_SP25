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

        # # Update integral and derivative terms
        self.integral += error * dt
        # derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.filtered_error = 0.9 * self.previous_error + 0.1 * error
        derivative = (self.filtered_error - self.previous_error) / dt if dt > 0 else 0.0

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        # output = self.Kp * error
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)

        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # Update previous values
        self.previous_error = error
        self.previous_time = current_time

        return output