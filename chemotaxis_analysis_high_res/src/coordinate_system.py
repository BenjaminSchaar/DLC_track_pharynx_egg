import pandas as pd

class CoordinateSystem:
    def __init__(self, top_left_pos, factor_px_to_mm=1.0, recording_type='crop', odor_pos=None):
        # Store the initial parameters
        self.top_left_x, self.top_left_y = top_left_pos
        self.recording_type = recording_type
        self.factor_px_to_mm = factor_px_to_mm

        # Store odor position if provided
        self.has_odor = odor_pos is not None
        if self.has_odor:
            self.odor_x, self.odor_y = odor_pos

            # Calculate odor position in mm for crop mode if in crop mode
            if recording_type == 'crop':
                self.odor_x_mm = (self.odor_y - self.top_left_y) * self.factor_px_to_mm
                self.odor_y_mm = (self.odor_x - self.top_left_x) * self.factor_px_to_mm
                # For consistency with the rotation in transform_coordinates_crop
                temp = self.odor_x_mm
                self.odor_x_mm = -self.odor_y_mm
                self.odor_y_mm = temp
                # Make X positive
                self.odor_x_mm = abs(self.odor_x_mm)

    def transform_coordinates(self, df):
        """
        Transform coordinates based on the recording type.
        """
        if self.recording_type == 'vid':
            return self.transform_coordinates_vid(df)
        else:  # Default to 'crop'
            return self.transform_coordinates_crop(df)

    def transform_coordinates_vid(self, df):
        """
        Transform coordinates to be non-negative and account for video-to-Matplotlib coordinate system differences.
        """
        # Determine the minimum values for X and Y (ensure all coordinates are positive)
        if self.has_odor:
            min_x = min(df['X'].min(), self.odor_x, self.top_left_x)
            min_y = min(df['Y'].min(), self.odor_y, self.top_left_y)
        else:
            min_x = min(df['X'].min(), self.top_left_x)
            min_y = min(df['Y'].min(), self.top_left_y)

        # Compute shifts to ensure all coordinates are non-negative
        shift_x = -min_x  # To move everything rightward
        shift_y = -min_y  # To move everything upward

        # Apply shifts to transform all coordinates into a positive coordinate space
        df['X_shifted'] = df['X'] + shift_x
        df['Y_shifted'] = df['Y'] + shift_y
        self.top_left_x_shifted = self.top_left_x + shift_x
        self.top_left_y_shifted = self.top_left_y + shift_y

        print(f"Applied X-shift: {shift_x}")
        print(f"Applied Y-shift: {shift_y}")

        # Transform to Matplotlib coordinates (swap X and Y, invert Y-axis)
        df['X_rel'] = abs(df['Y_shifted'] - self.top_left_y_shifted)  # Swap X and Y
        df['Y_rel'] = abs(-(df['X_shifted'] - self.top_left_x_shifted))  # Swap and flip Y-axis

        # Handle odor position if available
        if self.has_odor:
            self.odor_x_shifted = self.odor_x + shift_x
            self.odor_y_shifted = self.odor_y + shift_y

            # Calculate relative odor position
            self.odor_x_rel = abs(self.odor_y_shifted - self.top_left_y_shifted)  # Swap X and Y
            self.odor_y_rel = abs(-(self.odor_x_shifted - self.top_left_x_shifted))  # Swap and flip Y-axis

            print(f"Shifted odor position: x = {self.odor_x_shifted}, y = {self.odor_y_shifted}")
            print(f"Relative odor position: x = {self.odor_x_rel}, y = {self.odor_y_rel}")

            # Add transformed odor coordinates to DataFrame
            df['odor_x'] = self.odor_x_rel
            df['odor_y'] = self.odor_y_rel

        print(f"Shifted top-left position: x = {self.top_left_x_shifted}, y = {self.top_left_y_shifted}")
        print(f"Relative top-left position (should be 0,0): x = {0}, y = {0}")

        # Drop intermediate columns
        df = df.drop(['X_shifted', 'Y_shifted'], axis=1)

        return df

    def transform_coordinates_crop(self, df):
        """
        Transform coordinates exactly as in the Jupyter notebook implementation.
        """
        # Step 1: Convert X and Y to mm (note the swap in subtraction)
        df['X_mm'] = (df['X'] - self.top_left_y) * self.factor_px_to_mm
        df['Y_mm'] = (df['Y'] - self.top_left_x) * self.factor_px_to_mm

        # Step 2: Rotate X_mm and Y_mm by 90 degrees counterclockwise
        df['X_rel'] = -df['Y_mm']
        df['Y_rel'] = df['X_mm']

        # Step 3: Make X_mm_rotated positive
        df['X_rel'] = df['X_rel'].abs()

        # Remove intermediate columns
        df = df.drop(['X_mm', 'Y_mm'], axis=1)

        # Add rotated odor coordinates to DataFrame only if odor position was provided
        if self.has_odor:
            df['odor_x'] = self.odor_x_mm
            df['odor_y'] = self.odor_y_mm

        return df