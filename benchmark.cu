#define SIZE (256 * (1 << 20))  // 256 MB
#define NUM_RUNS 10

std::pair<float, float> measureBandwidth() {
    void* hostData;
    void* deviceData;

    hostData = malloc(SIZE);
    cudaMalloc(&deviceData, SIZE);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float totalHostToDeviceTime = 0.0f;
    float totalDeviceToHostTime = 0.0f;

    for (int run = 0; run < NUM_RUNS; ++run) {
        cudaEventRecord(startEvent, 0);
        cudaMemcpy(deviceData, hostData, SIZE, cudaMemcpyHostToDevice);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        float hostToDeviceTime;
        cudaEventElapsedTime(&hostToDeviceTime, startEvent, stopEvent);
        totalHostToDeviceTime += hostToDeviceTime;

        cudaEventRecord(startEvent, 0);
        cudaMemcpy(hostData, deviceData, SIZE, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        float deviceToHostTime;
        cudaEventElapsedTime(&deviceToHostTime, startEvent, stopEvent);
        totalDeviceToHostTime += deviceToHostTime;
    }

    float avgHostToDeviceBandwidth = (1000 * (float)SIZE * (float)NUM_RUNS) / (totalHostToDeviceTime * (float)(1 << 20));
    float avgDeviceToHostBandwidth = (1000 * (float)SIZE * (float)NUM_RUNS) / (totalDeviceToHostTime * (float)(1 << 20));

    free(hostData);
    cudaFree(deviceData);

    return std::make_pair(avgHostToDeviceBandwidth, avgDeviceToHostBandwidth);
}