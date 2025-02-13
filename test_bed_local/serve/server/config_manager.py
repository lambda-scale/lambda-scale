import os

from test_bed_local.serve.utils.utils import read_evaluation_parameters

params = read_evaluation_parameters()
root_path = params.get('root_path')

class ConfigManager:
    def __init__(self):
        config_file = f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/derecho_node.cfg'
        node_file = f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/node.cfg'
        self.config = self.read_config(config_file)
        self.my_id = int(self.config['my_id'])
        self.my_ip = self.config['my_ip']
        self.contact_ip = self.config['contact_ip']
        self.p2p_port = int(self.config['p2p_port'])
        self.total_p2p_nodes=int(self.config['total_p2p_nodes'])
        self.worker_log_file=self.config['worker_log_file']
        self.ctrl_log_file=self.config['ctrl_log_file']
        self.default_log_level=self.config['default_log_level']

        self.nodes_info = self.read_node_config(node_file)
        self.current_nodes=[]

    @staticmethod
    def read_config(filename):
        config = {}
        with open(filename, 'r') as file: 
            for line in file:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    config[key] = value
        return config
    
    def read_node_config(self, filename):
        """
        Reads the node configuration file and returns a dictionary of node information.
        """
        nodes_info = {}
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("#") or not line:  # Skip comments or empty lines
                    continue
                
                node_id, node_ip = line.split(",")
                node_id = int(node_id.strip())
                node_ip = node_ip.strip()
                
                nodes_info[node_id] = {
                    'my_id': node_id,
                    'my_ip': node_ip,
                    'p2p_port': self.p2p_port
                }
        return nodes_info
        
    
    def get_p2p_members(self):
        return sorted(self.nodes_info.keys())
        
    
    def get_p2p_view(self):
        p2p_view = {}
        for _,value in self.nodes_info.items():
            p2p_view[value['my_id']] = (value['my_ip'], value['p2p_port'])     
        return p2p_view
    
    def get_worker_log_name(self):
        return self.worker_log_file
    
    def get_ctrl_log_name(self):
        return self.ctrl_log_file
    
    def get_default_log_level(self):
        return self.default_log_level
        