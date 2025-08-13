import argparse
import pickle
import torch
import torch.optim as optim
import os


from utils.PINO3D_data import extract_data, create_physics_informed_data_loader
from utils.PINO3D_training import train, plot, test, tuning, plot_tuned
from models.FNO3D import physics_informed_FNO3D_Model
from models.WNO3D import physics_informed_WNO3D_Model
from models.Unet3D import physics_informed_Unet3D_Model
from models.U_FNO3D import physics_informed_UFNO3D_Model

# set arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'plot', 'plot_train', 'tune', 'plot_tuned'])
parser.add_argument('--model', type=str, default='FNO')
parser.add_argument('--data', type=str, default='homo3d')
parser.add_argument('--train_method', type=str, default='PINO3D')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--tuning_epochs', type=int, default=100)
args = parser.parse_args()

# load the data
if args.data == 'homo3d':
    with open('./simulation/data_general_incom_3D.pkl', 'rb') as handle:
        mat_contents = pickle.load(handle)
    data = extract_data(mat_contents)

train_loader, val_loader, test_loader = create_physics_informed_data_loader(data, [0.7,0.8], 1, train_shuffle=False)
print(len(train_loader), len(val_loader), len(test_loader))

# define devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'FNO':
    model = physics_informed_FNO3D_Model().float().to(device)
elif args.model == 'WNO':
    model = physics_informed_WNO3D_Model().float().to(device)
elif args.model == 'Unet':
    model = physics_informed_Unet3D_Model().float().to(device)
elif args.model == 'UFNO':
    model = physics_informed_UFNO3D_Model().float().to(device)

# try loading pre-trained model
try:
    model.load_state_dict(torch.load(r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method)))
    print('Loaded pre-trained model successfully.')
except:
    print('No pre-trained model found. Starting from scratch.')

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

if args.phase == 'plot_train':
    if args.data == 'homo3d':
        args.data = 'homo3d_train'
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
elif args.phase == 'tune':
    # Fine-tuning phase
    model = model.to(device)
    print('Starting fine-tuning phase...')
    args.epochs = args.tuning_epochs  # Use tuning epochs for fine-tuning
    tuned_model = tuning(args, model, device, (train_loader, val_loader, test_loader), optimizer)
    
    # Save the fine-tuned model
    os.makedirs('./trained_models', exist_ok=True)
    torch.save(tuned_model.state_dict(), r'./trained_models/{}_{}_{}_tuned.pth'.format(args.model, args.data, args.train_method))
    print('Fine-tuned model saved.')
    
    # Test the fine-tuned model
    err = test(args, tuned_model, device, test_loader)
    print('Fine-tuned model average relative error:', err)
elif args.phase == 'plot_tuned':
    # Plot comparison between original and fine-tuned models
    model = model.to(device)
    
    # Load fine-tuned model
    try:
        tuned_model = physics_informed_FNO3D_Model().float().to(device)
        tuned_model.load_state_dict(torch.load(r'./trained_models/{}_{}_{}_tuned.pth'.format(args.model, args.data, args.train_method)))
        print('Loaded fine-tuned model successfully.')
    except:
        print('No fine-tuned model found. Please run tuning phase first.')
        exit()
    
    plot_tuned(args, model, tuned_model, device, test_loader)
else:
    print(f'Unknown phase: {args.phase}')
    print('Available phases: train, test, plot, plot_train, tune, plot_tuned') 