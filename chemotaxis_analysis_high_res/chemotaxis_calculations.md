# **Documentation of Worm Tracking and Analysis Functions**

This document provides an overview of the functions used for worm tracking and analysis, detailing their purpose, parameters, returns, and the underlying mathematical/algorithmic logic.

---

## 1. `interpolate_df(vector, length) -> np.ndarray`

### **Purpose**
Interpolates a 1D array (vector) to a specified length, returning a new array with values obtained by linear interpolation.

### **Parameters**
- **vector** (`array-like`): The input vector (e.g., positions or any time-series data).
- **length** (`int`): The desired length of the output array after interpolation.

### **Returns**
- **np.ndarray**: The interpolated array of length `length`.

### **Logic and Mathematical Operations**
1. We create a new linearly spaced set of indices using:
   \[
     \text{new\_indices} = \linspace(0, \text{len}(vector) - 1, \text{length})
   \]
2. We interpolate the original `vector` using `np.interp()` at these new indices:
   \[
     \text{interpolated\_values} = \text{np.interp}(\text{new\_indices}, \text{arange}(\text{len}(vector)), vector)
   \]
3. The function returns the interpolated values as a NumPy array.

---

## 2. `correct_stage_pos_with_skeleton(...) -> pd.DataFrame`

### **Purpose**
Corrects the worm's position inside the arena based on:
- The worm's skeleton coordinates (spline) at a specified index.
- The recorded stage (or camera) position.
  
Optionally computes the centroid if `skel_pos = 100`.

### **Parameters**
- **worm_pos** (`pd.DataFrame`):  
  Contains columns `X_rel` and `Y_rel` — the worm’s relative positions in millimeters (after some conversion or tracking).
- **spline_X**, **spline_Y** (`pd.DataFrame`):  
  Contains the worm spline coordinates along the X and Y axes respectively. Typically each column represents a point along the worm's skeleton.
- **skel_pos** (`int`):  
  Skeleton index (0–99). If `skel_pos = 100`, then the centroid of all skeleton points is used.
- **video_resolution_x**, **video_resolution_y** (`int`):  
  Dimensions of the video in pixels.
- **factor_px_to_mm** (`float`):  
  Conversion factor to translate pixels to millimeters.
- **video_origin** (`str`):
  - `"video"`: Apply the original logic (swapped axes when subtracting differences).
  - `"crop"`: Use corrected logic (no axis swapping).

### **Returns**
- **pd.DataFrame**: The original `worm_pos` DataFrame with new columns for the corrected positions.

### **Logic and Mathematical Operations**
1. **Identify the Center of the Video**  
   \[
     \text{center\_x} = \frac{\text{video\_resolution\_x}}{2}, \quad
     \text{center\_y} = \frac{\text{video\_resolution\_y}}{2}
   \]

2. **Select Spline Coordinates**  
   - If `skel_pos = 100`, compute centroid along each row:
     \[
       \text{column\_skel\_pos\_x} = \text{spline\_X.mean(axis=1)}, \quad
       \text{column\_skel\_pos\_y} = \text{spline\_Y.mean(axis=1)}
     \]
   - Else, select column `skel_pos` from each DataFrame:
     \[
       \text{column\_skel\_pos\_x} = \text{spline\_X.iloc[:, skel\_pos]}, \quad
       \text{column\_skel\_pos\_y} = \text{spline\_Y.iloc[:, skel\_pos]}
     \]

3. **Compute Differences in Pixel Space**  
   \[
     \text{difference\_x\_px} = \text{column\_skel\_pos\_x} - \text{center\_x}, \quad
     \text{difference\_y\_px} = \text{column\_skel\_pos\_y} - \text{center\_y}
   \]

4. **Convert Pixel Differences to Millimeters**  
   \[
     \text{difference\_center\_x\_mm} = \text{difference\_x\_px} \times \text{factor\_px\_to\_mm}, \quad
     \text{difference\_center\_y\_mm} = \text{difference\_y\_px} \times \text{factor\_px\_to\_mm}
   \]

5. **Update `worm_pos` Based on `video_origin`**  
   - **If** `video_origin = "video"` (original logic, with axis swapping):
     \[
       \begin{cases}
       \text{X\_rel\_skel\_pos} = \text{X\_rel} - \text{difference\_center\_y\_mm} \\
       \text{Y\_rel\_skel\_pos} = \text{Y\_rel} - \text{difference\_center\_x\_mm}
       \end{cases}
     \]
   - **Else** `video_origin = "crop"` (corrected logic, no swapping):
     \[
       \begin{cases}
       \text{X\_rel\_skel\_pos} = \text{X\_rel} - \text{difference\_center\_x\_mm} \\
       \text{Y\_rel\_skel\_pos} = \text{Y\_rel} - \text{difference\_center\_y\_mm}
       \end{cases}
     \]

6. **Column Naming**  
   - When `skel_pos = 100`, the corrected positions are stored in `X_rel_skel_pos_centroid` and `Y_rel_skel_pos_centroid`.
   - Otherwise, the columns are named `X_rel_skel_pos_<skel_pos>` and `Y_rel_skel_pos_<skel_pos>`.

---

## 3. `calculate_distance(row: pd.Series, x_col: str, y_col: str, x_odor: float, y_odor: float) -> float`

### **Purpose**
Calculates the Euclidean distance from a worm position to the odor source on a row-by-row basis.

### **Parameters**
- **row** (`pd.Series`): A single row from a DataFrame.
- **x_col**, **y_col** (`str`): Column names in `row` that hold the worm’s x and y coordinates.
- **x_odor**, **y_odor** (`float`): Odor source coordinates (in the same coordinate system as `x_col` and `y_col`).

### **Returns**
- **float**: The Euclidean distance from `(x_rel, y_rel)` to `(x_odor, y_odor)`, or `np.nan` if coordinates are missing.

### **Logic and Mathematical Operations**
1. Extract the worm’s position from the row: \((x_{\text{rel}}, y_{\text{rel}})\).
2. If either `x_rel` or `y_rel` is `NaN`, return `NaN`.
3. Otherwise, compute:
   \[
     \text{distance} = \sqrt{(x_{\text{rel}} - x_{\text{odor}})^2 + (y_{\text{rel}} - y_{\text{odor}})^2}
   \]

---

## 4. `calculate_time_in_seconds(df: pd.DataFrame, fps: int) -> pd.DataFrame`

### **Purpose**
Adds a new column `"time_seconds"` to the DataFrame, which represents the time elapsed since the start of the recording, in seconds.

### **Parameters**
- **df** (`pd.DataFrame`): DataFrame containing a `"frame"` column or a numeric index.  
- **fps** (`int`): Frames per second of the recording.

### **Returns**
- **pd.DataFrame**: Original DataFrame with a new column `"time_seconds"`.

### **Logic and Mathematical Operations**
1. Check if `"frame"` column exists:
   - If **yes**, compute
     \[
       \text{df}["time\_seconds"] = \frac{\text{df}["frame"]}{\text{fps"}}
     \]
   - If **no**, use the index:
     \[
       \text{df}["time\_seconds"] = \frac{\text{index}}{\text{fps}}
     \]
2. This creates a continuous timeline in seconds from the frame or index number.

---

## 5. `calculate_preceived_conc(distance: float, time_seconds: float, conc_array: np.ndarray, distance_array: np.ndarray, diffusion_time_offset: int) -> float`

### **Purpose**
Derives a local concentration value (e.g., of an odor) based on:
- A simulated diffusion model stored in `conc_array`.
- A corresponding `distance_array` that maps indexes to radial distances.
- The worm’s current distance and recording time.

### **Parameters**
- **distance** (`float`): Current distance of the worm from the odor source.
- **time_seconds** (`float`): Current time in seconds of the worm’s recording.
- **conc_array**, **distance_array** (`np.ndarray`): 2D arrays representing the simulated concentration and distance values over time.
- **diffusion_time_offset** (`int`): Offset to shift the simulation time if the real experiment started after the diffusion was already in progress.

### **Returns**
- **float**: The interpolated (or nearest) concentration value at the worm’s position.

### **Logic and Mathematical Operations**
1. Convert time to an integer index (rounded) for accessing the simulation arrays:
   \[
     \text{sim\_time\_array} = \lfloor \text{time\_seconds} + 0.5 \rfloor + \text{diffusion\_time\_offset}
   \]
2. Retrieve the distance row from `distance_array` at `sim_time_array`:
   \[
     \text{distances\_in\_frame} = \text{distance\_array}[\text{sim\_time\_array}]
   \]
3. Find the closest distance in that row to the worm’s current distance:
   \[
     \text{closest\_distance} = \min(\text{distances\_in\_frame}, \text{ key=\lambda x: } |x - \text{distance}|)
   \]
4. Extract the index of that distance:
   \[
     \text{index\_of\_closest\_distance} = \text{np.where}(\text{distances\_in\_frame} == \text{closest\_distance})
   \]
5. Find the corresponding concentration from `conc_array` at the same time and distance index.
   \[
     \text{conc\_value} = \text{conc\_array}[\text{sim\_time\_array}][\text{index\_of\_closest\_distance}]
   \]

---

## 6. `calculate_displacement_vector(df_worm_parameter) -> pd.DataFrame`

### **Purpose**
Computes the displacement vector between consecutive frames (or rows), including:
- **dx_dt** and **dy_dt**: Rate of change in x and y coordinates.
- **displacement_vector_degrees**: Angle (bearing) of movement direction in degrees.
- **displacement_magnitude**: Length of the displacement vector.

### **Parameters**
- **df_worm_parameter** (`pd.DataFrame`): Must contain:
  - `X_rel_skel_pos_centroid`
  - `Y_rel_skel_pos_centroid`

### **Returns**
- **pd.DataFrame**: Original DataFrame with new columns added:
  - `centroid_dx_dt`
  - `centroid_dy_dt`
  - `centroid_displacement_vector_degrees`
  - `centroid_displacement_magnitude`

### **Logic and Mathematical Operations**
1. Extract `x` and `y`:
   \[
     x = \text{df["X\_rel\_skel\_pos\_centroid"].values}, \quad
     y = \text{df["Y\_rel\_skel\_pos\_centroid"].values}
   \]
2. Compute the discrete gradients (frame-to-frame differences):
   \[
     dx\_dt = \nabla x, \quad dy\_dt = \nabla y
   \]
   Using:
   \[
     \text{np.gradient}(x) \quad \text{and} \quad \text{np.gradient}(y)
   \]
3. Calculate the direction in radians:
   \[
     \text{direction\_radians} = \arctan2(dy\_dt, dx\_dt)
   \]
4. Convert to degrees:
   \[
     \text{direction\_degrees} = \left( \frac{180}{\pi} \right) \times \text{direction\_radians}
   \]
5. Calculate displacement magnitude:
   \[
     \text{displacement\_magnitude} = \sqrt{(dx\_dt)^2 + (dy\_dt)^2}
   \]

---

## 7. `calculate_curving_angle(df_worm_parameter, window_size=1) -> pd.DataFrame`

### **Purpose**
Determines the **curving angle**, i.e., how the bearing angle changes over a specified window of frames.

### **Parameters**
- **df_worm_parameter** (`pd.DataFrame`): Must have a column `centroid_displacement_vector_degrees`.
- **window_size** (`int`): How many frames to look back when calculating the angle change. Default is 1 (adjacent frame difference).

### **Returns**
- **pd.DataFrame**: Original DataFrame with a new `curving_angle` column.

### **Logic and Mathematical Operations**
1. Retrieve the array of displacement angles:
   \[
     \text{displacement\_vector\_degrees}
   \]
2. For each frame `i`, compute:
   \[
     \text{bearing\_change}[i] = \text{displacement\_vector\_degrees}[i] \;-\; \text{displacement\_vector\_degrees}[i - \text{window\_size}]
   \]
3. Normalize each difference to the range \([-180, 180]\):
   \[
     \text{bearing\_change}[i] = (\text{bearing\_change}[i] + 180) \% 360 - 180
   \]

---

## 8. `calculate_bearing_angle(df) -> pd.DataFrame`

### **Purpose**
Computes the worm’s **bearing angle** relative to the odor source, i.e., the difference between:
- The worm’s movement direction.
- The direction from the worm’s position to the odor source.

### **Parameters**
- **df** (`pd.DataFrame`): Must have:
  - `centroid_displacement_vector_degrees`
  - `X_rel_skel_pos_centroid`, `Y_rel_skel_pos_centroid`
  - `odor_x`, `odor_y`

### **Returns**
- **pd.DataFrame**: Original DataFrame with a new `bearing_angle` column.

### **Logic and Mathematical Operations**
1. Let
   \[
     \text{movement\_angle} = \text{df}["centroid\_displacement\_vector\_degrees"]
   \]
2. Compute the vector from worm to odor:
   \[
     x\_to\_odor = \text{odor\_x} - \text{X\_rel\_skel\_pos\_centroid}, \quad
     y\_to\_odor = \text{odor\_y} - \text{Y\_rel\_skel\_pos\_centroid}
   \]
3. Find the angle of that vector (in degrees):
   \[
     \text{angle\_to\_odor} = \arctan2(y\_to\_odor, x\_to\_odor) \times \frac{180}{\pi}
   \]
4. Bearing angle is the difference:
   \[
     \text{bearing\_angle} = \text{angle\_to\_odor} - \text{movement\_angle}
   \]
5. Normalize to \([-180, 180]\):
   \[
     \text{bearing\_angle} = ((\text{bearing\_angle} + 180) \% 360) - 180
   \]

---

## 9. `calculate_speed(df, fps) -> pd.DataFrame`

### **Purpose**
Calculates the instantaneous **speed** of the worm in units of (distance unit per second), given the displacement per frame.

### **Parameters**
- **df** (`pd.DataFrame`): Must contain `centroid_displacement_magnitude`.
- **fps** (`float`): Recording rate in frames per second.

### **Returns**
- **pd.DataFrame**: DataFrame with a new `speed` column.

### **Logic and Mathematical Operations**
1. The magnitude of displacement per frame is in the same units (e.g., mm/frame).
2. Multiply by `fps` to convert to (mm/second):
   \[
     \text{df["speed"]} = \text{df["centroid\_displacement\_magnitude"]} \times \text{fps}
   \]

---

## 10. `calculate_radial_speed(df, fps) -> pd.DataFrame`

### **Purpose**
Determines the **radial speed** (movement toward or away from the odor source) by looking at the gradient of the worm’s distance to the odor.

### **Parameters**
- **df** (`pd.DataFrame`): Must contain `distance_to_odor_centroid`.
- **fps** (`float`): Frames per second.

### **Returns**
- **pd.DataFrame**: DataFrame with a new `radial_speed` column.

### **Logic and Mathematical Operations**
1. Compute the gradient of the distance:
   \[
     \text{grad\_distance} = \nabla(\text{df}["distance\_to\_odor\_centroid"])
   \]
   This yields the distance difference between consecutive frames in the column.
2. Multiply by `fps` to convert from per-frame to per-second:
   \[
     \text{df}["radial\_speed"] = \text{grad\_distance} \times \text{fps}
   \]
3. Invert the sign so that moving **toward** the odor gives a **positive** radial speed, and **away** is negative:
   \[
     \text{df}["radial\_speed"] \leftarrow - \text{df}["radial\_speed"]
   \]

---

### **Notes & Best Practices**
- Ensure that all required columns exist in your DataFrames before applying these functions.  
- Adjust `fps` (frames per second) and `factor_px_to_mm` (pixel-to-mm conversion) according to the specifics of your experimental setup.
- For smoother analyses, you can apply rolling averages or other filtering methods to the displacement vectors and angles.

---

**End of Documentation**

