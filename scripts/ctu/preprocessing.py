import argparse
import ipaddress
import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Preprocess raw data')
parser.add_argument('--benign-data', type=str, required=True, help='path to benign data')
parser.add_argument('--malicious-data', type=str, required=True, help='path to malicious data')
parser.add_argument('--train-data', type=str, required=True, help='path to save the train data')
parser.add_argument('--test-data', type=str, required=True, help='path to save the test data')

args = parser.parse_args()

if not os.path.exists(args.benign_data) or not os.path.isfile(args.benign_data):
    sys.exit('Path to benign data does not exist or is not a file')

if not os.path.exists(args.malicious_data) or not os.path.isfile(args.malicious_data):
    sys.exit('Path to malicious data does not exist or is not a file')

ben_data = pd.read_csv(args.benign_data)

mal_data = pd.read_csv(args.malicious_data)

data = pd.concat([ben_data, mal_data], ignore_index=True)

data = data.drop('StartTime', axis=1)

data = data[data.Proto == 'tcp']
data.Proto = data.Proto.replace('tcp', 0)

data.sTos = data.sTos.replace(np.nan, 0.0)

data.dTos = data.dTos.replace(np.nan, 0.0)

data = data.dropna()

srcaddrport = data.SrcAddr.astype(str) + ':' + data.Sport.astype(str)
data.insert(0, 'SrcAddrPort', srcaddrport)

dstaddrport = data.DstAddr.astype(str) + ':' + data.Dport.astype(str)
data.insert(1, 'DstAddrPort', dstaddrport)

ipsrctype = data.SrcAddr.apply(lambda ip: 1 if (ipaddress.ip_address(ip) in ipaddress.ip_network('147.32.0.0/16')) else 0)
data.insert(2, 'IPSrcType', ipsrctype)
data = data.drop('SrcAddr', axis=1)

ipdsttype = data.DstAddr.apply(lambda ip: 1 if (ipaddress.ip_address(ip) in ipaddress.ip_network('147.32.0.0/16')) else 0)
data.insert(3, 'IPDstType', ipdsttype)
data = data.drop('DstAddr', axis=1)

sport = data.Sport.astype(int)
srcportwellknown = ((sport >= 0) & (sport <= 1023))
srcportwellknown = srcportwellknown.astype(int)
srcportregistered = ((sport >= 1024) & (sport <= 49151))
srcportregistered = srcportregistered.astype(int)
srcportprivate = (sport >= 49152)
srcportprivate = srcportprivate.astype(int)
data.insert(4, 'SrcPortWellKnown', srcportwellknown)
data.insert(5, 'SrcPortRegistered', srcportregistered)
data.insert(6, 'SrcPortPrivate', srcportprivate)
data = data.drop('Sport', axis=1)

dport = data.Dport.astype(int)
dstportwellknown = ((dport >= 0) & (dport <= 1023))
dstportwellknown = dstportwellknown.astype(int)
dstportregistered = ((dport >= 1024) & (dport <= 49151))
dstportregistered = dstportregistered.astype(int)
dstportprivate = (dport >= 49152)
dstportprivate = dstportprivate.astype(int)
data.insert(7, 'DstPortWellKnown', dstportwellknown)
data.insert(8, 'DstPortRegistered', dstportregistered)
data.insert(9, 'DstPortPrivate', dstportprivate)
data = data.drop('Dport', axis=1)

dir = pd.get_dummies(data.Dir)
dir.columns = ['->', '?>', '<?', '<?>']
data.insert(10, '->', dir['->'].astype(int))
data.insert(11, '?>', dir['?>'].astype(int))
data.insert(12, '<?', dir['<?'].astype(int))
data.insert(13, '<?>', dir['<?>'].astype(int))
data = data.drop('Dir', axis=1)

state = data.State.str.split('_', expand=True)
state.columns = ['SrcState', 'DstState']
data.insert(14, 'SrcStateA', state['SrcState'].str.contains('A', regex=False).astype(int))
data.insert(15, 'SrcStateC', state['SrcState'].str.contains('C', regex=False).astype(int))
data.insert(16, 'SrcStateE', state['SrcState'].str.contains('E', regex=False).astype(int))
data.insert(17, 'SrcStateF', state['SrcState'].str.contains('F', regex=False).astype(int))
data.insert(18, 'SrcStateP', state['SrcState'].str.contains('P', regex=False).astype(int))
data.insert(19, 'SrcStateR', state['SrcState'].str.contains('R', regex=False).astype(int))
data.insert(20, 'SrcStateS', state['SrcState'].str.contains('S', regex=False).astype(int))
data.insert(21, 'SrcStateU', state['SrcState'].str.contains('U', regex=False).astype(int))
data.insert(22, 'DstStateA', state['DstState'].str.contains('A', regex=False).astype(int))
data.insert(23, 'DstStateC', state['DstState'].str.contains('C', regex=False).astype(int))
data.insert(24, 'DstStateE', state['DstState'].str.contains('E', regex=False).astype(int))
data.insert(25, 'DstStateF', state['DstState'].str.contains('F', regex=False).astype(int))
data.insert(26, 'DstStateP', state['DstState'].str.contains('P', regex=False).astype(int))
data.insert(27, 'DstStateR', state['DstState'].str.contains('R', regex=False).astype(int))
data.insert(28, 'DstStateS', state['DstState'].str.contains('S', regex=False).astype(int))
data.insert(29, 'DstStateU', state['DstState'].str.contains('U', regex=False).astype(int))
data = data.drop('State', axis=1)

dstbytes = data.TotBytes - data.SrcBytes
data.insert(37, 'DstBytes', dstbytes)

bytesperpkt = data.TotBytes / data.TotPkts
data.insert(38, 'BytesPerPkt', bytesperpkt)

bytespersec = data.TotBytes / data.Dur
data.insert(39, 'BytesPerSec', bytespersec)
max = data.loc[data.BytesPerSec != np.inf, 'BytesPerSec'].max()
data.BytesPerSec = data.BytesPerSec.replace(np.inf, max)

pktspersec = data.TotPkts / data.Dur
data.insert(40, 'PktsPerSec', pktspersec)
max = data.loc[data.PktsPerSec != np.inf, 'PktsPerSec'].max()
data.PktsPerSec = data.PktsPerSec.replace(np.inf, max)

ratiooutin = data.SrcBytes / data.DstBytes
data.insert(41, 'RatioOutIn', ratiooutin)
max = data.loc[data.RatioOutIn != np.inf, 'RatioOutIn'].max()
data.RatioOutIn = data.RatioOutIn.replace(np.inf, max)

data = data[data.Dur < 300]
data = data[data.TotPkts < 100]
data = data[data.SrcBytes < 10000]
data = data[data.DstBytes < 60000]
data = data[data.BytesPerSec < 400000]
data = data[data.PktsPerSec < 10000]

ben_data = data[data.Label == 0]

mal_data = data[data.Label == 1]

if len(ben_data) >= len(mal_data) * 10:
    ben_data = ben_data.sample(len(mal_data) * 10)
else:
    mal_data = mal_data.sample(len(ben_data) // 10)

data = pd.concat([ben_data, mal_data], ignore_index=True).sample(frac=1)

train, test = train_test_split(data, test_size=0.2)

train.to_csv(args.train_data, index=False)
test.to_csv(args.test_data, index=False)