import os
import sys
from test_bed_local.serve.utils.utils import init_file_path, read_evaluation_parameters
root_path = str(sys.argv[1])
init_file_path(root_path)

import asyncio
import logging
from test_bed_local.serve.manager.manager import Manager

params = read_evaluation_parameters()
root_path = params.get('root_path')

filename = f'{root_path}/gpu-fast-scaling/test_bed_local/log/controller.log'
if os.path.exists(filename):
    os.remove(filename)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filename , level=logging.DEBUG)

async def main():
    manager = Manager()
    await manager.start()

if __name__ == "__main__":
    asyncio.run(main())