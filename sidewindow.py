import numpy as np
from numba import jit
# Implement first; optimize performance later
import tifffile, time, multiprocessing
import modifiedlabel_pool

def gaussian_kernel(size, sigma):
    """Generate a 3D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y, z: (1 / (2 * np.pi * sigma ** 2) ** 1.5) *
                        np.exp(-((x - (size - 1) / 2) ** 2 +
                                 (y - (size - 1) / 2) ** 2 +
                                 (z - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size, size)
    )
    return kernel

def extractCubesFromCube(cube, r, normalise):
    """
    Parameters
    ----------
    cube : source cube (3D array)
    r : kernel radius
    normalise : whether to normalize each extracted cube

    Returns
    -------
    Split the input cube according to the common SWF (side-window filter)
    specification used in the literature (eight sub-cubes).
    """
    cubes = []
    weights = []
    start_points = [
        (0, 0, 0),
        (0, 0, r),
        (0, r, 0),
        (0, r, r),
        (r, 0, 0),
        (r, 0, r),
        (r, r, 0),
        (r, r, r)
    ]

    for start_point in start_points:
        single_cube = cube[start_point[0]:start_point[0] + (r + 1),
                           start_point[1]:start_point[1] + (r + 1),
                           start_point[2]:start_point[2] + (r + 1)]
        if normalise:
            weights.append(np.sum(single_cube))
            single_cube = single_cube / np.sum(single_cube)
        cubes.append(single_cube)

    if normalise:
        weights = np.array(weights)
        return cubes, weights
    return cubes


# Helper functions
def weighted_average(weights, values, indices):
    """Compute weighted average for the given indices."""
    return sum(weights[i] * values[i] for i in indices) / sum(weights[i] for i in indices)


#@jit(nopython=True)
def applySWF3DdealBySlices(args):
    d_range, image_height, image_width, padded_image, r, kernals, weights, image, filtered_image = args
    # Iterate over all voxels [d, i, j]
    for d in d_range:
        for i in range(image_height):
            for j in range(image_width):
                # Extract local region centered at the target voxel.
                region = padded_image[d:d + 2*r + 1, i:i + 2*r + 1, j:j + 2*r + 1]
                # For side-window filtering, split the region using the same scheme as the kernel.
                regions_devided = extractCubesFromCube(region, r, normalise=False)

                # Convolve each kernel part with its corresponding region part.
                resulting_values = []
                # First handle the cubic kernels.
                for single_kernal, single_cube in zip(kernals, regions_devided):
                    v = np.sum(single_kernal * single_cube)
                    resulting_values.append(v)

                # Build rectangular kernels from weighted averages of cubic results.
                for ii in range(4):
                    resulting_values.append(weighted_average(weights, resulting_values, [ii, ii + 4]))

                for ii in range(4):
                    resulting_values.append(weighted_average(weights, resulting_values, [2 * ii, 2 * ii + 1]))

                for ii in [0, 1, 4, 5]:
                    resulting_values.append(weighted_average(weights, resulting_values, [ii, ii + 2]))

                # Choose the result closest to the original voxel intensity (L2 distance).
                L2distance = np.abs(resulting_values - image[d, i, j])
                filtered_image[d, i, j] = resulting_values[np.argmin(L2distance)]

    return filtered_image

@jit(nopython=True)
def applySWF3DdealBySlicesJIT(d_range, image_height, image_width, padded_image, r, kernals, weights, image, filtered_image):
    # Iterate over all voxels [d, i, j]
    for d in d_range:
        for i in range(image_height):
            for j in range(image_width):
                # Extract local region centered at the target voxel.
                region = padded_image[d:d + 2*r + 1, i:i + 2*r + 1, j:j + 2*r + 1]

                # Split region into 8 sub-cubes (same scheme as kernel).
                regions_devided = []
                start_points = [
                    (0, 0, 0),
                    (0, 0, r),
                    (0, r, 0),
                    (0, r, r),
                    (r, 0, 0),
                    (r, 0, r),
                    (r, r, 0),
                    (r, r, r)
                ]
                for start_point in start_points:
                    single_cube = region[start_point[0]:start_point[0] + (r + 1),
                                         start_point[1]:start_point[1] + (r + 1),
                                         start_point[2]:start_point[2] + (r + 1)]
                    regions_devided.append(single_cube)


                # Compute convolution results (store in a preallocated array for JIT).
                resulting_values = np.zeros(20)
                recorder = 0
                # First handle the cubic kernels.
                for single_kernal, single_cube in zip(kernals, regions_devided):
                    v = np.sum(single_kernal * single_cube)
                    resulting_values[recorder] = v
                    recorder += 1

                # Build rectangular kernels via weighted averages using precomputed weights.
                for ii in range(4):
                    resulting_values[recorder] = (weights[ii] * resulting_values[ii] + weights[ii + 4] * resulting_values[ii + 4]) / (weights[ii] + weights[ii + 4])
                    recorder += 1

                for ii in range(4):
                    resulting_values[recorder] = (weights[2 * ii] * resulting_values[2 * ii] + weights[2 * ii + 1] * resulting_values[2 * ii + 1]) / (weights[2 * ii] + weights[2 * ii + 1])
                    recorder += 1

                for ii in np.array([0, 1, 4, 5]):
                    resulting_values[recorder] = (weights[ii] * resulting_values[ii] + weights[ii + 2] * resulting_values[ii + 2]) / (weights[ii] + weights[ii + 2])
                    recorder += 1

                # Choose the result closest to the original voxel intensity (L2 distance).
                L2distance = np.abs(resulting_values - image[d, i, j])
                filtered_image[d, i, j] = resulting_values[np.argmin(L2distance)]

    return filtered_image

def USEapplySWF3DdealBySlicesJIT(args):
    d_range, image_height, image_width, padded_image, r, kernals, weights, image, filtered_image = args
    d_range = np.array(d_range)
    kernals = np.array(kernals)
    """
    Wrap and call the JIT-compiled function ensuring compatible array types.
    """
    return applySWF3DdealBySlicesJIT(d_range, image_height, image_width, padded_image, r, kernals, weights, image, filtered_image)

def applySWF3D(image, kernel, division=None, cpu_using=multiprocessing.cpu_count()-4, JIT=False):
    """
    Side-window filter (3D) implementation.

    Parameters
    ----------
    image : 3D numpy array representing a grayscale volume
    kernel : original convolution kernel (cube). Size must be odd.
    division : optional function to split kernel/region into parts
    cpu_using : number of CPU workers to use (default: cpu_count - 4)
    JIT : whether to use Numba JIT-compiled worker

    Returns
    -------
    filtered 3D volume (same shape as input)
    """

    # Basic parameters
    image_depth, image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    r = kernel_size // 2

    # Pad image borders to handle boundary voxels (reflect mode).
    padded_image = np.pad(image, ((r, r), (r, r), (r, r)), mode='reflect')

    # Output array
    filtered_image = np.zeros_like(image)

    # If no division function is provided, use the common SWF cube-to-rect partition.
    if division is None:
        # Split kernel into cubic parts (and compute weights if requested).
        kernals, weights = extractCubesFromCube(kernel, r, normalise=True)

        # Prepare execution: single-process or multi-process.
        if cpu_using == 1:  # run directly when only one CPU requested
            if JIT:
                filtered_image = USEapplySWF3DdealBySlicesJIT((range(image_depth), image_height, image_width, padded_image, r, kernals, weights, image, filtered_image))
            else:
                filtered_image = applySWF3DdealBySlices((range(image_depth), image_height, image_width, padded_image, r, kernals, weights, image, filtered_image))
        else:  # multi-core: distribute depth slices among workers
            # Split depth axis into slices for each worker.
            distribute_list = [i for i in range(0, image_depth + 1, image_depth // cpu_using)]
            if image_depth % cpu_using != 0:
                distribute_list.append(image_depth)

            # Run workers and collect results.
            if JIT:
                results = modifiedlabel_pool.run_multi_process(
                    USEapplySWF3DdealBySlicesJIT,
                    [(range(distribute_list[start], distribute_list[start + 1], 1),
                      image_height, image_width,
                      padded_image, r, kernals,
                      weights, image, filtered_image)
                     for start in range(len(distribute_list) - 1)]
                )
            else:
                results = modifiedlabel_pool.run_multi_process(
                    applySWF3DdealBySlices,
                    [(range(distribute_list[start], distribute_list[start + 1], 1),
                      image_height, image_width, padded_image, r, kernals, weights, image, filtered_image)
                     for start in range(len(distribute_list) - 1)]
                )

            # Sum per-slice results to obtain final output.
            for result in results:
                filtered_image = filtered_image + result

    # If a custom division function is provided, use it (no cube-to-rect optimization).
    else:
        kernels = division(kernel, r)
        for d in range(image_depth):
            for i in range(image_height):
                for j in range(image_width):
                    region = padded_image[d:d + 2*r + 1, i:i + 2*r + 1, j:j + 2*r + 1]
                    regions_devided = division(region, r)
                    resulting_values = []
                    # Handle cubic parts
                    for single_kernal, single_cube in zip(kernels, regions_devided):
                        v = np.sum(single_kernal * single_cube)
                        resulting_values.append(v)
                    L2distance = np.abs(resulting_values - image[d, i, j])
                    filtered_image[d, i, j] = resulting_values[np.argmin(L2distance)]

    return filtered_image


# Example usage
if __name__ == "__main__":
    # This small example shows how to call the side-window filter (`applySWF3D`) on a synthetic volume.
    import time as _time
    np.random.seed(0)
    depth, height, width = 7, 7, 7
    image = np.random.rand(depth, height, width).astype(np.float32)
    kernel_size = 3
    kernel = gaussian_kernel(kernel_size, sigma=1.0)

    # Run filter in single-process mode for demonstration.
    t0 = _time.time()
    filtered = applySWF3D(image, kernel, division=None, cpu_using=1, JIT=False)
    t1 = _time.time()
    print("Input shape:", image.shape, "Filtered shape:", filtered.shape, f"Elapsed: {t1 - t0:.4f}s")

    # Optionally save the result if tifffile is available.
    try:
        tifffile.imwrite("sidewindow_example_output.tif", filtered.astype(image.dtype))
        print("Saved output to sidewindow_example_output.tif")
    except Exception as _e:
        print("Could not save output (tifffile missing or error):", _e)

