import argparse
import pickle
import torch
import torch.optim as optim
import os


from utils.PINO3D_data import extract_data, create_one_sample_data_loader
from utils.PINO3D_training import plot_tuned, tuning
from models.FNO3D import physics_informed_FNO3D_Model

# set arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='tune', choices=['tune', 'plot'])
parser.add_argument('--model', type=str, default='FNO')
parser.add_argument('--data', type=str, default='homo3d')
parser.add_argument('--train_method', type=str, default='PINO3D')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--sample_id', type=int, default=49)
args = parser.parse_args()

# load the data
if args.data == 'homo3d':
    with open('./simulation/data_general_incom_3D.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)
elif args.data == 'heter3d':
    with open('./simulation/data_general_heter_3D.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)

loader = create_one_sample_data_loader(data, args.sample_id)

# define devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'FNO':
    model = physics_informed_FNO3D_Model().float().to(device)
    model_tuned = physics_informed_FNO3D_Model().float().to(device)
else:
    print(f'Model {args.model} not implemented for 3D yet.')
    exit()

# try loading pre-trained model
try:
    model.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
    model_tuned.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
    print('Loaded pre-trained model successfully.')
except:
    print('No pre-trained model found. Please train a model first.')
    exit()

# define optimizer
optimizer = optim.Adam(model_tuned.parameters(), lr=0.0001)

# exp
if args.phase == 'tune':
    print('Starting physics-informed fine-tuning...')
    model = model.to(device)
    model_tuned = model_tuned.to(device)
    model_tuned = tuning(args, model_tuned, device, (loader, loader, loader), optimizer)
    
    # Save the fine-tuned model
    os.makedirs('./trained_models', exist_ok=True)
    torch.save(model_tuned.state_dict(), r'./trained_models/{}_{}_{}_tuned.pth'.format(args.model, args.data, args.train_method))
    print('Fine-tuned model saved.')
    
elif args.phase == 'plot':
    model = model.to(device)
    model_tuned = model_tuned.to(device)
    
    # Try to load fine-tuned model if it exists
    try:
        model_tuned.load_state_dict(torch.load(r'./trained_models/{}_{}_{}_tuned.pth'.format(args.model, args.data, args.train_method)))
        print('Loaded fine-tuned model successfully.')
    except:
        print('No fine-tuned model found. Running fine-tuning first...')
        model_tuned = tuning(args, model_tuned, device, (loader, loader, loader), optimizer)
        
        # Save the fine-tuned model
        os.makedirs('./trained_models', exist_ok=True)
        torch.save(model_tuned.state_dict(), r'./trained_models/{}_{}_{}_tuned.pth'.format(args.model, args.data, args.train_method))
        print('Fine-tuned model saved.')
    
    # Plot comparison
    args.data = f'{args.data}_tuned'
    plot_tuned(args, model, model_tuned, device, loader)
else:
    print(f'Unknown phase: {args.phase}')
    print('Available phases: tune, plot') 