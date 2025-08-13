import argparse
import pickle
import torch
import torch.optim as optim


from utils.PINO_data import extract_data, extract_3D_slice_data, create_physics_informed_data_loader
from utils.PINO_training import train, plot, test
from models.FNO import physics_informed_FNO_Model
from models.Unet import physics_informed_Unet_Model
from models.U_FNO import physics_informed_UFNO_Model
from models.WNO import physics_informed_WNO_Model

# set arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='test')
parser.add_argument('--model', type=str, default='WNO')
parser.add_argument('--data', type=str, default='heter')
parser.add_argument('--train_method', type=str, default='PINO')
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# load the data
if args.data == 'heter':
    with open('./simulation/data_general_heter.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)
elif args.data == 'homo':
    with open('./simulation/data_general_homo.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)
elif args.data == 'homo3d_slicing':
    with open('./simulation/data_general_incom_3D.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_3D_slice_data(mat_contents)

train_loader, val_loader, test_loader = create_physics_informed_data_loader(data, [0.7,0.8], 1, train_shuffle=False)
print(len(train_loader), len(val_loader), len(test_loader))

# define devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'FNO':
    model =  physics_informed_FNO_Model()
elif args.model == 'Unet':
    model = physics_informed_Unet_Model()
elif args.model == 'UFNO':
    model = physics_informed_UFNO_Model()
elif args.model == 'WNO':
    model = physics_informed_WNO_Model()

# try loading pre-trained model
try:
    model.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
except:
    print('No pre-trained model.')

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if args.phase == 'plot_train':
    if args.data == 'heter':
        args.data = 'heter_train'
    if args.data == 'homo':
        args.data = 'homo_train'
    test_loader = train_loader

# exp
if args.phase == 'train':
    model = model.to(device)
    train(args, model, device, (train_loader, val_loader, test_loader), optimizer)
elif args.phase == 'test':
    model = model.to(device)
    err = test(args, model, device, test_loader)
    print('average relative error:', err)
elif args.phase == 'plot' or args.phase == 'plot_train':
    model = model.to(device)
    plot(args, model, device, test_loader)
