# mount_drift — sinew-mocap

A hexagon cluster (`core/` + `ports/` + `adapters/`) of the Sinew mocap stack:
the TIC calibrator — recovers each sensor's mount/drift via a trained net, and hosts the shared vkc Vulkan compute backend.

## Build

```
cmake -B build && cmake --build build
# tests: cmake -B build -DSINEW_TESTS=ON && cmake --build build && ctest --test-dir build
```

## Dependencies

See `ports/sibling-repos.txt` — clone the listed `sinew-mocap` repos side-by-side in `~/`.
