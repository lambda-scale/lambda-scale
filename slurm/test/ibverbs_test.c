#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#define PORT 7471
#define BUFFER_SIZE 4096

// Error checking macros
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE); } } while (0)
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(EXIT_FAILURE); } } while (0)

// RDMA Connection Context
struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    char *buffer;   // CUDA device buffer
};

// Initialize RDMA resources
struct rdma_context* init_rdma_context() {
    struct rdma_context *rdma = malloc(sizeof(*rdma));
    memset(rdma, 0, sizeof(*rdma));

    // Get list of devices
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    CHECK(dev_list);

    // Open the first device
    rdma->ctx = ibv_open_device(dev_list[0]);
    CHECK(rdma->ctx);

    // Allocate Protection Domain (PD)
    rdma->pd = ibv_alloc_pd(rdma->ctx);
    CHECK(rdma->pd);

    // Allocate CUDA device buffer and register memory
    CUDA_CHECK(cudaMalloc((void**)&rdma->buffer, BUFFER_SIZE));
    rdma->mr = ibv_reg_mr(rdma->pd, rdma->buffer, BUFFER_SIZE,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    CHECK(rdma->mr);

    // Create Completion Queue (CQ)
    rdma->cq = ibv_create_cq(rdma->ctx, 10, NULL, NULL, 0);
    CHECK(rdma->cq);

    return rdma;
}

// Clean up RDMA resources
void cleanup_rdma_context(struct rdma_context *rdma) {
    if (!rdma) return;
    if (rdma->qp) ibv_destroy_qp(rdma->qp);
    if (rdma->cq) ibv_destroy_cq(rdma->cq);
    if (rdma->mr) ibv_dereg_mr(rdma->mr);
    if (rdma->pd) ibv_dealloc_pd(rdma->pd);
    if (rdma->ctx) ibv_close_device(rdma->ctx);
    if (rdma->buffer) CUDA_CHECK(cudaFree(rdma->buffer));
    free(rdma);
}

// Initialize Queue Pair (QP)
void init_qp(struct rdma_context *rdma) {
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = rdma->cq,
        .recv_cq = rdma->cq,
        .cap = {
            .max_send_wr = 10,
            .max_recv_wr = 10,
            .max_send_sge = 1,
            .max_recv_sge = 1,
        },
        .qp_type = IBV_QPT_RC,
    };

    rdma->qp = ibv_create_qp(rdma->pd, &qp_attr);
    CHECK(rdma->qp);
}

// Server: Listen and accept connection
void run_server() {
    struct rdma_context *rdma = init_rdma_context();
    init_qp(rdma);

    printf("Server is listening on port %d...\n", PORT);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(sock != -1);

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = INADDR_ANY,
    };

    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 1);

    int client_sock = accept(sock, NULL, NULL);
    CHECK(client_sock != -1);

    // Receive remote QP information
    uint32_t remote_qp_num;
    uint32_t remote_lid;
    read(client_sock, &remote_qp_num, sizeof(remote_qp_num));
    read(client_sock, &remote_lid, sizeof(remote_lid));

    printf("Received QP Number: %u, LID: %u\n", remote_qp_num, remote_lid);

    // Post a receive request
    struct ibv_sge sge = {
        .addr = (uintptr_t)rdma->buffer,
        .length = BUFFER_SIZE,
        .lkey = rdma->mr->lkey,
    };

    struct ibv_recv_wr recv_wr = {
        .wr_id = 1,
        .sg_list = &sge,
        .num_sge = 1,
    }, *bad_recv_wr;

    CHECK(ibv_post_recv(rdma->qp, &recv_wr, &bad_recv_wr) == 0);
    printf("Server is ready to receive data.\n");

    // Wait for completion
    struct ibv_wc wc;
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);
    CHECK(wc.status == IBV_WC_SUCCESS);
    printf("Received message: %s\n", rdma->buffer);

    // Cleanup
    close(client_sock);
    close(sock);
    cleanup_rdma_context(rdma);
}

// Client: Connect to server and send data
void run_client(const char *server_ip) {
    struct rdma_context *rdma = init_rdma_context();
    init_qp(rdma);

    printf("Connecting to server at %s:%d...\n", server_ip, PORT);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(sock != -1);

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = inet_addr(server_ip),
    };

    CHECK(connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != -1);

    // Query the port to get the LID
    struct ibv_port_attr port_attr;
    CHECK(ibv_query_port(rdma->ctx, 1, &port_attr) == 0); // Port 1 is typically used
    uint32_t lid = port_attr.lid;

    // Send QP information to the server
    uint32_t qp_num = rdma->qp->qp_num;
    write(sock, &qp_num, sizeof(qp_num));
    write(sock, &lid, sizeof(lid));

    // Perform RDMA Write
    strcpy(rdma->buffer, "Hello, RDMA with CUDA!");
    struct ibv_sge sge = {
        .addr = (uintptr_t)rdma->buffer,
        .length = BUFFER_SIZE,
        .lkey = rdma->mr->lkey,
    };

    struct ibv_send_wr send_wr = {
        .wr_id = 1,
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
    }, *bad_send_wr;

    CHECK(ibv_post_send(rdma->qp, &send_wr, &bad_send_wr) == 0);

    // Wait for completion
    struct ibv_wc wc;
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);
    CHECK(wc.status == IBV_WC_SUCCESS);

    printf("Client sent data: %s\n", (char*)rdma->buffer);

    close(sock);
    cleanup_rdma_context(rdma);
}

// Main function
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <server|client> [server_ip]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "server") == 0) {
        run_server();
    } else if (strcmp(argv[1], "client") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Client mode requires server IP.\n");
            return EXIT_FAILURE;
        }
        run_client(argv[2]);
    } else {
        fprintf(stderr, "Invalid mode. Use 'server' or 'client'.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
