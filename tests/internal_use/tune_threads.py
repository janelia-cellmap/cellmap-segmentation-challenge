# %%
# Tune threading
import sys
from time import sleep, perf_counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from tqdm import tqdm
from cellmap_segmentation_challenge.utils import TEST_CROPS
from cellmap_segmentation_challenge.evaluate import INSTANCE_CLASSES

# THESE ARE MADE UP VALUES AND MAY NEED ADJUSTING TO ACCURATELY REFLECT THE COMPUTATIONAL COSTS
match_time_ratio = 1e-90  # s/voxel
cost_time_ratio = 1e-50  # s/voxel
dist_time_ratio = 1e-70  # s/voxel
semantic_time_ratio = 1e-60  # s/voxel
precompute_time_ratio = 1000  # s/voxel
cost_loop_ratio = 1e-10  # iters/voxel
dist_loop_ratio = 1e-30  # iters/voxel


def time_instance(size, instance_threads, precompute_limit):
    # spoof cost calculation
    if size > precompute_limit:
        sleeper = lambda: sleep(size * cost_time_ratio * precompute_time_ratio)
    else:
        sleeper = lambda: sleep(size * cost_time_ratio)

    loops = int(size * cost_loop_ratio)
    with ThreadPoolExecutor(max_workers=instance_threads) as instance_pool:
        results = instance_pool.map(sleeper, range(loops))

    # spoof distance calculation
    sleeper = lambda: sleep(size * dist_time_ratio)
    loops = int(size * dist_loop_ratio)
    with ThreadPoolExecutor(max_workers=instance_threads) as instance_pool:
        results = instance_pool.map(sleeper, range(loops))


def time_label(volume, instance_threads, precompute_limit):
    label, size = volume
    # spoof matching crop space
    sleep(size * match_time_ratio)

    if label in INSTANCE_CLASSES:
        time_instance(size, instance_threads, precompute_limit)
    else:
        # spoof semantic scoring
        sleep(size * semantic_time_ratio)


def time_volume(volume, label_threads, instance_threads, precompute_limit):
    partial_func = partial(
        time_label, instance_threads=instance_threads, precompute_limit=precompute_limit
    )
    with ThreadPoolExecutor(max_workers=label_threads) as label_pool:
        results = label_pool.map(partial_func, volume)


def time_submission(
    main_threads=2, label_threads=4, instance_threads=4, precompute_limit=1e9
):
    volumes = {}
    for crop in TEST_CROPS:
        if crop.dataset not in volumes:
            volumes[crop.dataset] = []
        volumes[crop.dataset].append([crop.class_label, np.prod(crop.shape)])

    partial_func = partial(
        time_volume,
        label_threads=label_threads,
        instance_threads=instance_threads,
        precompute_limit=precompute_limit,
    )
    with ThreadPoolExecutor(max_workers=main_threads) as main_pool:
        results = main_pool.map(partial_func, volumes.values())


# %timeit time_submission(4, 4, 4, 1e6)

# %%
# TUNE THREADS
main_threads = [1, 2, 4, 8]
label_threads = [1, 2, 4, 8]
instance_threads = [1, 2, 4, 8]
precompute_limits = [1e6, 1e7, 1e8, 1e9]
results = np.empty(
    (
        len(main_threads),
        len(label_threads),
        len(instance_threads),
        len(precompute_limits),
    )
)
with tqdm(total=np.prod(results.shape)) as pbar:
    for i, main_thread in enumerate(main_threads):
        for j, label_thread in enumerate(label_threads):
            for k, instance_thread in enumerate(instance_threads):
                for l, p in enumerate(precompute_limits):
                    start = perf_counter()
                    time_submission(main_thread, label_thread, instance_thread, p)
                    end = perf_counter()
                    results[i, j, k, l] = end - start
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "main": main_thread,
                            "label": label_thread,
                            "instance": instance_thread,
                            "precompute": p,
                        }
                    )
                    sys.stdout.flush()  # Ensures the progress bar updates
                    # print(
                    #     f"main: {main_thread}, label: {label_thread}, instance: {instance_thread}, precompute: {p}, time: {results[i, j, k, l]}"
                    # )
# %%
# What's the best threading configuration?
best = np.unravel_index(np.argmin(results), results.shape)
print(
    f"Best threading configuration: main: {main_threads[best[0]]}, label: {label_threads[best[1]]}, instance: {instance_threads[best[2]]}, precompute: {precompute_limits[best[3]]: .0e}"
)
print(f"Time: {results[best]}")

# %%
