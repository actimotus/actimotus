import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AutoCalibrate:
    window: str | timedelta = '10s'
    tolerance: float = 0.015
    samples: int = 50
    calib_cube: float = 0.3
    max_iter = 1_000
    improv_tol = 0.0004
    err_tol = 0.01

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # 1. Setup
        window = pd.Timedelta(self.window)
        cols = ['acc_x', 'acc_y', 'acc_z']

        # 2. Resample and identify stationary periods
        # Aggregate mean and SD in one pass
        resampled = df[cols].resample(window, origin='start').agg(['mean', 'std'])

        # Extract sub-frames for cleaner filtering
        means = resampled.xs('mean', axis=1, level=1)
        stds = resampled.xs('std', axis=1, level=1)

        # Filter: all axes must have SD < tolerance
        is_stationary = (stds < self.tolerance).all(axis=1)  # type: ignore
        stationary_means = means[is_stationary].dropna().to_numpy()  # type: ignore

        # 3. Filter Zero Vectors (Artifacts)
        # Norms must be significantly larger than 0 to avoid div/0 errors later
        norms = np.linalg.norm(stationary_means, axis=1)
        valid_norms = norms > 1e-8
        stationary_means = stationary_means[valid_norms]

        if len(stationary_means) < self.samples:
            logger.warning(
                f'Calibration skipped: Not enough stationary data ({len(stationary_means)}/{self.samples} samples).'
            )
            return df

        # 4. Check calibration cube (sphere) coverage
        # Ensure enough data points on the sphere
        min_vals = stationary_means.min(axis=0)
        max_vals = stationary_means.max(axis=0)

        if (max_vals < self.calib_cube).any() or (min_vals > -self.calib_cube).any():
            logger.warning('Calibration skipped: Data does not cover the full calibration sphere.')
            return df

        # 5. Initialization
        # Keep an immutable copy of the raw stationary measurements
        raw_stationary = stationary_means.copy()
        current = raw_stationary.copy()

        # Calculate initial target (vectors)
        target = current / np.linalg.norm(current, axis=1, keepdims=True)
        errors = np.linalg.norm(current - target, axis=1)
        init_error = np.mean(errors)

        if init_error < self.err_tol:
            logger.info(f'Data already well-calibrated (Error: {init_error:.4f}).')
            return df

        # 6. Iterative optimization
        offset = np.zeros(3)
        scale = np.ones(3)

        best_offset = offset.copy()
        best_scale = scale.copy()
        best_error = init_error

        for i in range(self.max_iter):
            # 99.5th percentile error to filter outliers
            max_error_quantile = np.quantile(errors, 0.995)

            if max_error_quantile == 0:
                logger.info('Perfect calibration achieved.')
                break

            # Calculate weights (bi-square weight function roughly)
            weights = np.maximum(1 - errors / max_error_quantile, 0)
            sqrt_weights = np.sqrt(weights)

            # Solve weighted least squares for each axis independently
            # Model: target_k = scale_k * (raw_k + offset_k)
            # Linearize as: target_k = p1*raw_k + p0
            for k in range(3):
                regressor = current[:, k]
                outcome = target[:, k]

                # Construct weighted design matrix [1, x] * w
                A_w = np.column_stack((np.ones_like(regressor), regressor)) * sqrt_weights[:, None]
                y_w = outcome * sqrt_weights

                # Solve A_w * params = y_w
                params, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=None)
                p0, p1 = params[0], params[1]

                # Update global accumulated coefficients
                # new_offset = p0 + p1 * old_offset
                # new_scale = p1 * old_scale
                offset[k] = p0 + p1 * offset[k]
                scale[k] = p1 * scale[k]

            # Apply new coefficients to ORIGINAL raw data
            current = offset + (raw_stationary * scale)

            # Re-normalize targets and calc errors
            target = current / np.linalg.norm(current, axis=1, keepdims=True)
            errors = np.linalg.norm(current - target, axis=1)
            mean_error = np.mean(errors)

            # Check improvement
            if mean_error < best_error:
                error_improvement = (best_error - mean_error) / best_error
                best_error = mean_error
                best_offset = offset.copy()
                best_scale = scale.copy()

                if error_improvement < self.improv_tol:
                    logger.debug(f'Converged at iteration {i} (improvement < tolerance).')
                    break
            else:
                # If error increased - likely overshot or hit noise floor
                logger.debug(f'Optimization stopped at iteration {i} (error increased).')
                break

        # 7. Finalize
        if best_error > self.err_tol:
            logger.warning(
                f'Calibration finished but did not meet tolerance. Error: {init_error:.4f} -> {best_error:.4f}.'
            )
        else:
            logger.info(
                f'Calibration successful. Error: {init_error:.4f} -> {best_error:.4f}. '
                f'Offset: {np.round(best_offset, 3)}, Scale: {np.round(best_scale, 3)}.'
            )

        # Apply best coefficients to the full dataset
        calibrated = best_offset + (df[cols].to_numpy() * best_scale)

        # Reconstruct DataFrame preserving index
        calibrated = pd.DataFrame(calibrated, columns=cols, index=df.index, dtype=np.float32)

        return calibrated
