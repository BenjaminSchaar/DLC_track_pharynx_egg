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
        if self.recording_type == 'zim01':
            return self.transform_coordinates_zim01(df)
        elif self.recording_type == 'zim06':
            return self.transform_coordinates_zim06(df)
        else:  # Default to 'crop'
            return self.transform_coordinates_crop(df)

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

    def transform_coordinates_zim01(self, df):
        """
        Transform coordinates so that the top-left corner becomes the origin (0,0),
        and relative coordinates increase positively to the right and upward
        (by swapping axes after flipping relative to the top-left corner).
        """
        df = df.copy()

        # Flip signs so that coordinates are relative to top-left
        x_rel = self.top_left_x - df['X']
        y_rel = self.top_left_y - df['Y']

        # Swap axes so Y increases upwards
        df['X_rel'] = y_rel
        df['Y_rel'] = x_rel

        print(f"Using top-left position as reference: x = {self.top_left_x}, y = {self.top_left_y}")
        print(f"Relative top-left position (should be 0,0): x = 0, y = 0")

        # Handle odor position if available
        if self.has_odor:
            odor_x_rel = self.top_left_x - self.odor_x
            odor_y_rel = self.top_left_y - self.odor_y

            self.odor_x_rel = odor_y_rel  # Swapped
            self.odor_y_rel = odor_x_rel  # Swapped

            print(f"Original odor position: x = {self.odor_x}, y = {self.odor_y}")
            print(f"Relative odor position: x = {odor_y_rel}, y = {odor_x_rel}")

            df['odor_x'] = odor_y_rel
            df['odor_y'] = odor_x_rel

        return df

    def transform_coordinates_zim06(self, df):
        """
        Transform coordinates so that the top-left corner becomes the origin (0,0),
        and ensure all coordinates are positive.
        """
        df = df.copy()

        # Flip signs so that coordinates are relative to top-left
        x_rel = df['X'] - self.top_left_x
        y_rel = df['Y'] - self.top_left_y

        # Swap axes so Y increases upwards
        df['X_rel'] = y_rel
        df['Y_rel'] = x_rel

        # Make coordinates positive by taking absolute value - simplest approach
        df['X_rel'] = df['X_rel'].abs()
        df['Y_rel'] = df['Y_rel'].abs()

        print(f"Using top-left position as reference: x = {self.top_left_x}, y = {self.top_left_y}")

        # Handle odor position if available
        if self.has_odor:
            odor_x_rel = self.odor_x - self.top_left_x
            odor_y_rel = self.odor_y - self.top_left_y

            self.odor_x_rel = odor_y_rel  # Swapped
            self.odor_y_rel = odor_x_rel  # Swapped

            # Make odor coordinates positive too
            self.odor_x_rel = abs(self.odor_x_rel)
            self.odor_y_rel = abs(self.odor_y_rel)

            print(f"Original odor position: x = {self.odor_x}, y = {self.odor_y}")
            print(f"Relative odor position: x = {self.odor_x_rel}, y = {self.odor_y_rel}")

            df['odor_x'] = self.odor_x_rel
            df['odor_y'] = self.odor_y_rel

        return df