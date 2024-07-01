__kernel void vector_add(
    const float a,
    __global const float *x,
    __global const float *y,
    __global float *out)
{
    int id = get_global_id(0);
    out[id] = a * x[id] + y[id];
}
