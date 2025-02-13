#!bash/bin

rm -rf /client_bin && \
mkdir /client_bin && \
cp /app/build-client/src/client/librtclient.so /client_bin && \
cd /client_bin && \
ldd librtclient.so | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' . && \
cp /app/build-client/src/client/libmycuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \

cp /app/tests/multi_thread_test.py /
echo "LD_LIBRARY_PATH=/client_bin/:\$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so CLIENT_ID=\$1 CUR_SERVER_ID=\$2 python \$3 \$4" > /start_with_server_id.sh 
bash /start_with_server_id.sh {0} "$1" multi_thread_test.py {9000}