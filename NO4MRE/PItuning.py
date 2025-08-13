import argparse
import pickle
import torch
import torch.optim as optim


from utils.PINO_data import extract_data, create_one_sample_data_loader
from utils.PINO_training import plot_tuned, tuning
from models.FNO import physics_informed_FNO_Model
from models.Unet import physics_informed_Unet_Model
from models.U_FNO import physics_informed_UFNO_Model
from models.WNO import physics_informed_WNO_Model

# set arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='plot')
parser.add_argument('--model', type=str, default='FNO')
parser.add_argument('--data', type=str, default='heter')
parser.add_argument('--train_method', type=str, default='PINO')
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

# load the data
if args.data == 'heter':
    with open('/taiga/illinois/eng/cee/meidani/Vincent/MRE/data_general_heter.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)
elif args.data == 'homo':
    with open('/taiga/illinois/eng/cee/meidani/Vincent/MRE/data_general_incom.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)
loader = create_one_sample_data_loader(data, 49)

# define devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'FNO':
    model =  physics_informed_FNO_Model()
    model_tuned =  physics_informed_FNO_Model()
elif args.model == 'Unet':
    model = physics_informed_Unet_Model()
elif args.model == 'UFNO':
    model = physics_informed_UFNO_Model()
elif args.model == 'WNO':
    model = physics_informed_WNO_Model()

# try loading pre-trained model
try:
    model.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
    model_tuned.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
except:
    print('No pre-trained model.')

# define optimizer
optimizer = optim.Adam(model_tuned.parameters(), lr=0.0001)

# exp
model = model.to(device)
model_tuned = model_tuned.to(device)
model_tuned = tuning(args, model_tuned, device, (loader, loader, loader), optimizer)
args.data = 'heter_tuned'
plot_tuned(args, model, model_tuned, device, loader)
