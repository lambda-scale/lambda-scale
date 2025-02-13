
K=2
node_group_block_group = [[]for _ in range(K)]
node_group = []

node_num = 32
block_num = 12

for node_group_id in range(K):
    init = node_group_id
    while init < block_num:
        node_group_block_group[node_group_id].append(init)
        init += K

num_per_node_group = node_num // K
num_per_block_group = block_num // K
node_arith = node_num  % K
block_arith = block_num  % K

for i in range(K):
    node_id_begin = (i * num_per_node_group + min(i, node_arith)) + 1 
    node_id_end = ((i + 1) * num_per_node_group + min(i + 1, node_arith)) + 1 
    node_group.append((node_id_begin,node_id_end))

def get_K_from_node_id(node_id):
    k1 = (node_id-1) // num_per_node_group + 1
    k2 = ((node_id-1) // (num_per_node_group+1))
    for k in range(k2,k1):
        if node_id >= node_group[k][0] and node_id <node_group[k][1]:
            return k

def is_final_id(node_id,block_id):
    node_group_id = get_K_from_node_id(node_id)
    final_node_group_id = 0
    if node_group_id == 0:
        final_node_group_id = K-1
    else:
        final_node_group_id = node_group_id-1
    final_id = node_group_block_group[final_node_group_id][-1]

    if final_id == block_id:
        return True
    else:
        return False

def next_block_id(node_id,block_id):
    if is_final_id(node_id,block_id):
        return None
    node_group_id = block_id%K
    if block_id == node_group_block_group[node_group_id][-1]:
        next_node_group_id = (node_group_id+1)%K
        return node_group_block_group[next_node_group_id][0]
    else:
        return block_id+K

def generate_block_id_order_by_K_contrast(k):
    list = []
    for i in range(0,block_num):
        list.append(i)
    return list

def generate_block_id_order_by_K(k):
    list = []
    for i in range(k,K):
        list+=node_group_block_group[i]
    for i in range(k):
        list+=node_group_block_group[i]
    return list
print(node_group_block_group[0])
print(generate_block_id_order_by_K(1))
print(next_block_id(18,10))