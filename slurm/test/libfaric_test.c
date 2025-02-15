#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT "7471"   // Port for communication
#define BUF_SIZE 1024 // Buffer size for transfer

void gpu_setup(void **gpu_buffer, size_t size) {
    cudaError_t cuda_err;

    // Allocate GPU buffer
    cuda_err = cudaMalloc(gpu_buffer, size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }
    printf("Allocated GPU buffer: %p\n", *gpu_buffer);

    // Initialize GPU buffer
    cuda_err = cudaMemset(*gpu_buffer, 0, size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA memset failed: %s\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }
}

void run_server() {
    struct fi_info *hints, *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fid_eq *eq;
    struct fid_cq *cq;
    struct fid_mr *mr;
    struct fi_cq_attr cq_attr = {0};
    struct fi_eq_attr eq_attr = {0};
    void *gpu_buffer;

    // Setup GPU buffer
    gpu_setup(&gpu_buffer, BUF_SIZE);

    // Initialize hints
    hints = fi_allocinfo();
    hints->ep_attr->type = FI_EP_MSG;  // Reliable Connection endpoint
    hints->caps = FI_MSG;    // Messaging and RMA
    hints->mode = 0;          // Required mode for `libfabric`
    hints->fabric_attr->prov_name = strdup("verbs"); // Force IB provider
    hints->domain_attr->mr_mode = FI_MR_LOCAL;

    // Get fabric info
    int ret = fi_getinfo(FI_VERSION(1, 11), NULL, PORT, FI_SOURCE, hints, &info);
    if (ret) {
        perror("fi_getinfo");
        fprintf(stderr, "fi_getinfo failed: %s\n", fi_strerror(-ret));
        fprintf(stderr, "Requested capabilities: 0x%llx\n", hints->caps);
        fprintf(stderr, "Requested modes: 0x%llx\n", hints->mode);
        fprintf(stderr, "Requested MR mode: 0x%llx\n", hints->domain_attr->mr_mode);
        exit(EXIT_FAILURE);
    }

    // Open fabric
    if (fi_fabric(info->fabric_attr, &fabric, NULL)) {
        perror("fi_fabric");
        exit(EXIT_FAILURE);
    }

    // Open domain
    if (fi_domain(fabric, info, &domain, NULL)) {
        perror("fi_domain");
        exit(EXIT_FAILURE);
    }

    // Create endpoint
    if (fi_endpoint(domain, info, &ep, NULL)) {
        perror("fi_endpoint");
        exit(EXIT_FAILURE);
    }

    // Create event queue
    eq_attr.size = 256;
    if (fi_eq_open(fabric, &eq_attr, &eq, NULL)) {
        perror("fi_eq_open");
        exit(EXIT_FAILURE);
    }

    // Bind endpoint to event queue
    if (fi_ep_bind(ep, &eq->fid, 0)) {
        perror("fi_ep_bind");
        exit(EXIT_FAILURE);
    }

    // Create completion queue
    cq_attr.size = 256;
    cq_attr.format = FI_CQ_FORMAT_DATA;
    if (fi_cq_open(domain, &cq_attr, &cq, NULL)) {
        perror("fi_cq_open");
        exit(EXIT_FAILURE);
    }

    // Bind endpoint to completion queue
    if (fi_ep_bind(ep, &cq->fid, FI_RECV | FI_TRANSMIT)) {
        perror("fi_ep_bind (CQ)");
        exit(EXIT_FAILURE);
    }

    // Register GPU buffer
    if (fi_mr_reg(domain, gpu_buffer, BUF_SIZE, FI_RECV | FI_SEND, 0, 0, 0, &mr, NULL)) {
        perror("fi_mr_reg");
        exit(EXIT_FAILURE);
    }

    // Enable endpoint
    if (fi_enable(ep)) {
        perror("fi_enable");
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on port %s...\n", PORT);

    // Wait for connection
    struct fi_eq_cm_entry cm_entry;
    uint32_t event;
    ssize_t rd;
    if ((rd = fi_eq_sread(eq, &event, &cm_entry, sizeof(cm_entry), -1, 0)) < 0) {
        perror("fi_eq_sread");
        exit(EXIT_FAILURE);
    }

    if (event != FI_CONNECTED) {
        fprintf(stderr, "Unexpected event: %u\n", event);
        exit(EXIT_FAILURE);
    }

    printf("Client connected!\n");

    // Receive data
    struct fi_cq_data_entry cq_entry;
    while (1) {
        if ((rd = fi_cq_read(cq, &cq_entry, 1)) > 0) {
            printf("Received data on GPU buffer\n");
        } else if (rd < 0 && rd != -FI_EAGAIN) {
            perror("fi_cq_read");
            break;
        }
    }

    // Clean up
    fi_close(&ep->fid);
    fi_close(&cq->fid);
    fi_close(&mr->fid);
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    cudaFree(gpu_buffer);
}

void run_client(const char *server_ip) {
    struct fi_info *hints, *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fid_cq *cq;
    struct fid_mr *mr;
    struct fi_cq_attr cq_attr = {0};
    void *gpu_buffer;

    // Setup GPU buffer
    gpu_setup(&gpu_buffer, BUF_SIZE);
    strcpy((char *)gpu_buffer, "Hello from GPU!");

    // Initialize hints
    hints = fi_allocinfo();
    hints->ep_attr->type = FI_EP_MSG;  // Reliable Connection endpoint
    hints->caps = FI_MSG | FI_RMA;     // Messaging and RMA
    hints->mode = FI_CONTEXT;          // Required mode for `libfabric`
    hints->fabric_attr->prov_name = strdup("verbs"); // Force IB provider

    // Get fabric info
    if (fi_getinfo(FI_VERSION(1, 11), server_ip, PORT, 0, hints, &info)) {
        perror("fi_getinfo");
        exit(EXIT_FAILURE);
    }

    // Open fabric
    if (fi_fabric(info->fabric_attr, &fabric, NULL)) {
        perror("fi_fabric");
        exit(EXIT_FAILURE);
    }

    // Open domain
    if (fi_domain(fabric, info, &domain, NULL)) {
        perror("fi_domain");
        exit(EXIT_FAILURE);
    }

    // Create endpoint
    if (fi_endpoint(domain, info, &ep, NULL)) {
        perror("fi_endpoint");
        exit(EXIT_FAILURE);
    }

    // Create completion queue
    cq_attr.size = 256;
    cq_attr.format = FI_CQ_FORMAT_DATA;
    if (fi_cq_open(domain, &cq_attr, &cq, NULL)) {
        perror("fi_cq_open");
        exit(EXIT_FAILURE);
    }

    // Bind endpoint to completion queue
    if (fi_ep_bind(ep, &cq->fid, FI_RECV | FI_TRANSMIT)) {
        perror("fi_ep_bind (CQ)");
        exit(EXIT_FAILURE);
    }

    // Register GPU buffer
    if (fi_mr_reg(domain, gpu_buffer, BUF_SIZE, FI_RECV | FI_SEND, 0, 0, 0, &mr, NULL)) {
        perror("fi_mr_reg");
        exit(EXIT_FAILURE);
    }

    // Enable endpoint
    if (fi_enable(ep)) {
        perror("fi_enable");
        exit(EXIT_FAILURE);
    }

    printf("Connecting to server at %s...\n", server_ip);

    // Connect to server
    if (fi_connect(ep, info->dest_addr, NULL, 0)) {
        perror("fi_connect");
        exit(EXIT_FAILURE);
    }

    printf("Connected to server!\n");

    // Send data
    if (fi_send(ep, gpu_buffer, BUF_SIZE, NULL, 0, NULL)) {
        perror("fi_send");
        exit(EXIT_FAILURE);
    }

    printf("Data sent from GPU buffer\n");

    // Clean up
    fi_close(&ep->fid);
    fi_close(&cq->fid);
    fi_close(&mr->fid);
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    cudaFree(gpu_buffer);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <role: server|client> [server_ip]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "server") == 0) {
        run_server();
    } else if (strcmp(argv[1], "client") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Error: Client requires server_ip\n");
            return EXIT_FAILURE;
        }
        run_client(argv[2]);
    } else {
        fprintf(stderr, "Invalid role: %s. Use 'server' or 'client'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
