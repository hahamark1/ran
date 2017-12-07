import subprocess
#
# cmd = ['python', 'main.py', '--cuda', '--emsize', '256', '--nhid', '1024', '--dropout'
#         , '0.5', '--epochs', '60', '--nlayers', '1', '--batch-size', '40', '--model', 'LSTM', '--save', 'LSTM-1layer.pt', '--data1', 'LSTM-1layer']
#
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# process.wait()
# for line in process.stdout:
#     print(line)
#
# cmd = ['python', 'main.py', '--cuda', '--emsize', '256', '--nhid', '1024', '--dropout'
#         , '0.5', '--epochs', '60', '--nlayers', '1', '--batch-size', '40', '--model', 'RAN', '--save', 'RAN-1layer.pt', '--data1', 'RAN-3layer']
#
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# process.wait()
# for line in process.stdout:
#     print(line)
#
# cmd = ['python', 'main.py', '--cuda', '--emsize', '256', '--nhid', '1024', '--dropout'
#         , '0.5', '--epochs', '60', '--nlayers', '2', '--batch-size', '40', '--model', 'LSTM', '--save', 'LSTM-2layer.pt', '--data1', 'LSTM-2layer']
#
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# process.wait()
# for line in process.stdout:
#     print(line)

cmd = ['python', 'main.py', '--cuda', '--emsize', '256', '--nhid', '1024', '--dropout'
        , '0.5', '--epochs', '28', '--nlayers', '3', '--batch-size', '40', '--model', 'LSTM', '--save', 'LSTM-3layer.pt', '--data1', 'LSTM-3layer']

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.wait()
for line in process.stdout:
    print(line)

cmd = ['python', 'main.py', '--cuda', '--emsize', '256', '--nhid', '1024', '--dropout'
        , '0.5', '--epochs', '60', '--nlayers', '5', '--batch-size', '40', '--model', 'LSTM', '--save', 'LSTM-5layer.pt', '--data1', 'LSTM-5layer']

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.wait()
for line in process.stdout:
    print(line)
