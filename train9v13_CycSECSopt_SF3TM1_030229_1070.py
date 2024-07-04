#MB:02.06.31-1080--02.07.13--02.07.16---train5: 02.09.01
"""
Trains MaskCycleGAN-VC as described in https://arxiv.org/pdf/2102.12841.pdf
Inspired by https://github.com/jackaduma/CycleGAN-VC2
MB_Moradi:02.05.20.1080--02.07.13--02.07.28---02.08.03---02.08.12 train2.py--02.08.15-mean,var of realA for fakeA wav generation and Mean var of B for fakeB
#MB:03.01.22 IQA Immmage similarity Metric.Opt or Image Quality Aware
#MB:03.01.23 complete IQA Opt. unified by metricGAN structure

train5v00_ConvMask-SF3TM1_030203_1080.py--03.02.03
train6v00_ConvMask_SF3TM1_030204_1080.py
train6v01_ConvMask_SF3TF1_030204_1080.py
train6v11_CycMCDopt_SF3TF1_030204_1080.py :03.02.04

train7v22_CycSTOIopt_SF3TF1_030210_1080.PY :03.02.10
train8v32_CycSTOIopt_SM3TF1_030216_1080.py 03.02.16
train9v32_CycSTOIopt_SM3TF1_030217_1080.py 03.02.17
train9v20_ConvMetricVC_SF3TF1_030218_1070.py  03.02.17
train9v10_ConvMetricVC_SF3TM1_030220_1070.py  03.02.20
train9v12_CycSTOIopt_SF3TM1_030221_1070.py  03.02.21
train9v13_CycSECSopt_SF3TM1_030229_1070.py   03.02.29

"""

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.utils import decode_melspectrogram, get_mel_spectrogram_fig
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver
#mb02.05.03:IMPORTs
import torchaudio
import soundfile as sf

#from pystoi import stoi
import mystoi
from mystoi import stoi
from resemblyzer import VoiceEncoder, preprocess_wav
resemb_encoder = VoiceEncoder()

#from pathlib import Path

from pypesq import pesq
from mymcd import Calculate_MCD
from nisqa.NISQA_model import nisqaModel
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

import librosa
import math

import pyworld
import pysptk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

#import numpy as np # DEIM
#import torch # DEIM

from torch import optim
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import imageio
#MB:02.11.26
from torch.autograd import Variable

from IQA_pytorch import SSIM, MS_SSIM, CW_SSIM, GMSD, LPIPSvgg, DISTS, NLPD, FSIM, VSI, VIFs, VIF, MAD

from scipy.signal import fftconvolve, hann


class MaskCycleGANVCTraining(object):
    """Trainer for MaskCycleGAN-VCs
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store args
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.stop_identity_after = args.stop_identity_after
        self.mini_batch_size = args.batch_size
        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.epochs_per_plot = args.epochs_per_plot

        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        
        # TODO add proper argument for evaluation dataset path
        # loading dataset for test process
        
        # Initialize speaker's  datasets
        
        #eval_path = './vcc2018_preprocessed/vcc2018_evaluation_smaller' #TODO revise
        #eval_path = './vcc2018_preprocessed/vcc2018_evaluation_new20' #TODO revise
        
        #eval_path = './vcc2018_preprocessed/newsmall_evaluation020525' #TODO revi
        """
        # Initialize speakerA's train dataset
        self.test_dataset_A = self.loadPickleFile(os.path.join(
            './vcc2018_preprocessed/newsmall_evaluation020525' , args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
        test_dataset_A_norm_stats = np.load(os.path.join(
            './vcc2018_preprocessed/newsmall_evaluation020525' , args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
        self.test_dataset_A_mean = test_dataset_A_norm_stats['mean']
        self.test_dataset_A_std = test_dataset_A_norm_stats['std']

        # Initialize speakerB's test dataset
        self.test_dataset_B = self.loadPickleFile(os.path.join(
            './vcc2018_preprocessed/newsmall_evaluation020525' , args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
        test_dataset_B_norm_stats = np.load(os.path.join(
            './vcc2018_preprocessed/newsmall_evaluation020525', args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
        self.test_dataset_B_mean = test_dataset_B_norm_stats['mean']
        self.test_dataset_B_std = test_dataset_B_norm_stats['std']
        
        """
        
        #-------------------------------------------Test & Evaluation Data for 1080-------------------------------------------#
        orig_evaluation_data = True
        #vad_evaluation_data = True
        print("\norig_evaluation_data=",orig_evaluation_data)
        
        if (orig_evaluation_data==True):
            print("\nUsed Data preprocessed for Eval: **Original VCC2018 Data**")
            #eval_path = './vcc2018_preprocessed/orig_evaluation' #TODO revise

            # Initialize speakerA's test dataset
            self.test_dataset_A = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed_orig64fr/vcc2018_evaluation' , args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
            test_dataset_A_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed_orig64fr/vcc2018_evaluation', args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
            self.test_dataset_A_mean = test_dataset_A_norm_stats['mean']
            self.test_dataset_A_std = test_dataset_A_norm_stats['std']

            # Initialize speakerB's test dataset
            self.test_dataset_B = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed_orig64fr/vcc2018_evaluation' , args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
            test_dataset_B_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed_orig64fr/vcc2018_evaluation', args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
            self.test_dataset_B_mean = test_dataset_B_norm_stats['mean']
            self.test_dataset_B_std = test_dataset_B_norm_stats['std']

        else:
            print("\nUsed Data preprocessed for Eval: **preprocessed_128fr_bigtr_smalleval**")
            #eval_path='/mnt/main/voice_conversion/MaskCycleGAN-VC/vcc2018_preprocessed_128fr_bigtr_smalleval/vcc2018_evaluation'
        
            # Initialize speakerA's test dataset
            self.test_dataset_A = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed_128fr_bigtr_smalleval/vcc2018_evaluation' , args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
            test_dataset_A_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed_128fr_bigtr_smalleval/vcc2018_evaluation', args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
            self.test_dataset_A_mean = test_dataset_A_norm_stats['mean']
            self.test_dataset_A_std = test_dataset_A_norm_stats['std']

            # Initialize speakerB's test dataset
            self.test_dataset_B = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed_128fr_bigtr_smalleval/vcc2018_evaluation' , args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
            test_dataset_B_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed_128fr_bigtr_smalleval/vcc2018_evaluation', args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
            self.test_dataset_B_mean = test_dataset_B_norm_stats['mean']
            self.test_dataset_B_std = test_dataset_B_norm_stats['std']
            
        vad_evaluation_data = False
        if (vad_evaluation_data==True):
            print("\nUsed Data preprocessed for Eval: **Original VCC2018 Data**")
            #eval_path = './vcc2018_preprocessed/orig_evaluation' #TODO revise

            # Initialize speakerA's train dataset
            self.test_dataset_A = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed/vad_evaluation' , args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
            test_dataset_A_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed/vad_evaluation', args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
            self.test_dataset_A_mean = test_dataset_A_norm_stats['mean']
            self.test_dataset_A_std = test_dataset_A_norm_stats['std']

            # Initialize speakerB's test dataset
            self.test_dataset_B = self.loadPickleFile(os.path.join(
                './vcc2018_preprocessed/vad_evaluation' , args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
            test_dataset_B_norm_stats = np.load(os.path.join(
                './vcc2018_preprocessed/vad_evaluation', args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
            self.test_dataset_B_mean = test_dataset_B_norm_stats['mean']
            self.test_dataset_B_std = test_dataset_B_norm_stats['std']  
        #------------------------------------------------Test & Evaluation Data for 1080-------------------------------------#

        # Initialize speakerA's train dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        # Initialize speakerB's train dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        
        # Compute lr decay rate
        self.n_samples = len(self.dataset_A)
        print(f'n_samples = {self.n_samples}')
        self.generator_lr_decay = self.generator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        self.discriminator_lr_decay = self.discriminator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        print(f'generator_lr_decay = {self.generator_lr_decay}')
        print(f'discriminator_lr_decay = {self.discriminator_lr_decay}')

        
        self.test_dataset = VCDataset(datasetA=self.test_dataset_A,
                                    datasetB=self.test_dataset_B,
                                    n_frames=args.num_frames_validation,
                                    max_mask_len=args.max_mask_len,
                                    valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 drop_last=False)
                
        # Initialize Train Dataloader
        self.num_frames = args.num_frames
        self.dataset = VCDataset(datasetA=self.dataset_A,
                                 datasetB=self.dataset_B,
                                 n_frames=args.num_frames,
                                 max_mask_len=args.max_mask_len)
        

        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                            batch_size=self.mini_batch_size,
                                                            shuffle=True,
                                                            drop_last=False)

        # Initialize Validation Dataloader (used to generate intermediate outputs)
        self.validation_dataset = VCDataset(datasetA=self.dataset_A,
                                            datasetB=self.dataset_B,
                                            n_frames=args.num_frames_validation,
                                            max_mask_len=args.max_mask_len,
                                            valid=True)
        self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 drop_last=False)

        # Initialize logger and saver objects
        self.logger = TrainLogger(args, len(self.train_dataloader.dataset))
        self.saver = ModelSaver(args)

        # Initialize Generators and Discriminators
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_A2 = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_B2 = Discriminator().to(self.device)

        # Initialize Optimizers
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters()) + \
            list(self.discriminator_A2.parameters()) + \
            list(self.discriminator_B2.parameters())
        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # Load from previous ckpt
        if args.continue_train:
            self.saver.load_model(
                self.generator_A2B, "generator_A2B", None, self.generator_optimizer)
            self.saver.load_model(self.generator_B2A,
                                  "generator_B2A", None, None)
            self.saver.load_model(self.discriminator_A,
                                  "discriminator_A", None, self.discriminator_optimizer)
            self.saver.load_model(self.discriminator_B,
                                  "discriminator_B", None, None)
            self.saver.load_model(self.discriminator_A2,
                                  "discriminator_A2", None, None)
            self.saver.load_model(self.discriminator_B2,
                                  "discriminator_B2", None, None)

    def adjust_lr_rate(self, optimizer, generator):
        """Decays learning rate.

        Args:
            optimizer (torch.optim): torch optimizer
            generator (bool): Whether to adjust generator lr.
        """
        if generator:
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        """Sets gradients of the generators and discriminators to zero before backpropagation.
        """
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def train(self):
        """Implements the training loop for MaskCycleGAN-VC
        """
        
        #MB:02.04.13
        # path1='/mnt/main/voice_conversion/MaskCycleGAN-VC/vcc2018/vcc2018_training_bigger/VCC2SF3/'
        # path2='/mnt/main/voice_conversion/MaskCycleGAN-VC/vcc2018/vcc2018_training_bigger/VCC2TM1/'
        # path3='/mnt/main/voice_conversion/MaskCycleGAN-VC/vcc2018_preprocessed/vcc2018_training_bigger/VCC2SF3/'
        # path3='/mnt/main/voice_conversion/MaskCycleGAN-VC/vcc2018_preprocessed/vcc2018_training_bigger/VCC2TM1/'
        # MB:02.02.03
                
        #MB:02.03.13
        #text01= " [MMCD_GTB Scores A-B(0-1): ConvertedB-TargetB ] for A2B path:\n"       
        # file01 = open("report_files/MMCD_GTB_scores.txt", "w")    
        # #file1.write(text3)
        # file01.close()
        with open('report_files/MMCD_GBTA.txt', 'w')  as f01 :
            pass
        
        # #MB:02.04.17
        # #text02= "  for A2B path:\n"       
        # file02 = open("report_files/realA_genB_stoi.txt", "w")    
        # #file02.write(text04)
        # file02.close() 
        with open('report_files/realA_genB_stoi.txt', 'w') as f02 :
            pass
        
        # #text3 = "cosinesim of converted and target of A2B path:\n"
        # file03 = open('report_files/cosinesim_conBtarg.txt', 'w')
        # # Writing a string to file
        # #file03.write(text1)
        # file03.close()
        with open('report_files/cosinesim_genBtarg.txt', 'w') as f03 :
             pass
        
        #MB:02.02.13
        #text04 = " [MONISQA SCORE A-B(0-1): 5.0-(Target Nisqa - Converted Nisqa)/5.0 ] for A2B path:\n"
        #text04 = " [MONISQA SCORE A-B(0-1):  Converted Nisqa/5.0 for A2B path:\n"
        #mb:02.03.26
        # file04 = open("report_files/ Nisqatts_GB_norm.txt", "w")
        # #file04.write(text2)
        # file04.close()
        with open('report_files/Nisqatts_GB_norm.txt', 'w') as f04 :
             pass
            
        with open('report_files/realA_genB_tr_pesqnorm.txt', 'w') as f05 :
             pass
            
        with open('report_files/realA_genB_tr_sdrnorm.txt', 'w') as f06 :
             pass
        ##############################################################
        with open('report_files/realB_fakeB_test_dtwStoi.txt', 'w') as f11 :
             pass
        
        
        """ TODO delete
        #MB:02.02.09
        file11='vcc2018_preprocessed/vcc2018_training_bigger/VCC2SF3/VCC2SF3_norm_stat.npz'
        with open(file11, 'rb') as f11:  
                _= f11.seek(0) 
                npzfile_A=np.load(f11)  
                #########print("\n*@npzfile_A.files",npzfile_A.files)
                self.dataset_A_mean=npzfile_A['mean']
                self.dataset_A_std=npzfile_A['std']
                #f4.close()
        #print("\n*@self.dataset_A_mean",self.dataset_A_mean)
        #print("\n*@self.dataset_A_std",self.dataset_A_std)
        
        file12='vcc2018_preprocessed/vcc2018_training_bigger/VCC2TM1/VCC2TM1_norm_stat.npz'
        with open(file12, 'rb') as f12:  
                _ = f12.seek(0)
                npzfile_B=np.load(f12)  
                #############print("\n*@npzfile_B.files",npzfile_B.files)
                self.dataset_B_mean=npzfile_B['mean']
                self.dataset_B_std=npzfile_B['std']
                #f12.close()
        """
        #MB:02.03.10
        #As a Representative of Training Source Speaker:
        alpha_norm= torch.load( 'report_files/alpha_norm.pt').cuda()
        #print("Loading the file:'alpha_norm.npy' for alpha_norm")
        #As a Representative of Training Target Speaker:
        betta_norm= torch.load( 'report_files/betta_norm.pt').cuda()
        #As a Representative of Source Evaluation Speaker:
        landa_norm= torch.load( 'report_files/landa_norm.pt').cuda()
        #As a Representative of Target Evaluation(Reference) Speaker:
        gamma_norm= torch.load( 'report_files/gamma_norm.pt').cuda()
        
        # model_gru = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder',
        #                        pretrained = True)   
        # model_convgru = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder',
        #                        pretrained = True)    
        
    
        #MB:02.08.13
        #As a Representative of Training Source Speaker by Resemblyzer:
        trainSF3spkEmbed = torch.load( 'report_files/trainSF3spkEmbed.pt').cuda()
        #As a Representative of Training Target Speaker:
        #trainTM1spkEmbed = torch.load( 'report_files/trainTF1spkEmbed.pt').cuda() Error02.09.01
        trainTM1spkEmbed = torch.load( 'report_files/trainTM1spkEmbed.pt').cuda()
        #As a Representative of Source Evaluation Speaker:
        evalSF3spkEmbed  = torch.load(  'report_files/evalSF3spkEmbed.pt').cuda()
        #As a Representative of Target Evaluation(Reference) Speaker:
        #evalTM1spkEmbed  = torch.load(  'report_files/evalTF1spkEmbed.pt').cuda() Error02.09.01
        evalTM1spkEmbed  = torch.load(  'report_files/evalTM1spkEmbed.pt').cuda() 
        
    
        #MB:02.08.13
        #As a Representative of Training Source Speaker by Resemblyzer:
        #trainSM3spkEmbed = torch.load( 'report_files/trainSF3spkEmbed.pt').cuda()Error02.09.01
        trainSM3spkEmbed = torch.load( 'report_files/trainSM3spkEmbed.pt').cuda()
        #As a Representative of Training Target Speaker:
        trainTF1spkEmbed = torch.load( 'report_files/trainTF1spkEmbed.pt').cuda()
        #As a Representative of Source Evaluation Speaker:
        #evalSM3spkEmbed  = torch.load(  'report_files/evalSF3spkEmbed.pt').cuda() Error02.09.01
        evalSM3spkEmbed  = torch.load(  'report_files/evalSM3spkEmbed.pt').cuda()
        #As a Representative of Target Evaluation(Reference) Speaker:
        evalTF1spkEmbed  = torch.load(  'report_files/evalTF1spkEmbed.pt').cuda()
        
        
        # MMCD metrics objects
        MCD_mode_train = "dtw" # Another option is "plain"
        
        if (MCD_mode_train == "dtw"): 
            #maxpymcd_train = 20
            maxpymcd_train =  25 ##02.08.12 for SF3-TM1 train2.py ok
            
        else: 
            #maxpymcd_train = 25
            maxpymcd_train = 30
        print("MCD_mode_train is:",MCD_mode_train,"maxpymcd_train=: ",maxpymcd_train)
            
        mcd_toolbox_train = Calculate_MCD(MCD_mode = MCD_mode_train)
        
        
        MCD_mode_test = "dtw"  # Another option is "plain"
        
        if (MCD_mode_test == "dtw"): 
            #maxpymcd_test = 20
            maxpymcd_test = 15
        else: 
            #maxpymcd_test = 25
            maxpymcd_test = 20
        
        mcd_toolbox_test = Calculate_MCD(MCD_mode = MCD_mode_test)
        
        
        # NISQA metrics objects
        
        nisqa_args_train = {
            'mode'            : 'predict_dir',
            'pretrained_model': 'nisqa/weights/nisqa_tts.tar',
            'data_dir'        : 'report_files/wavs/train/',
            'num_workers'     : 0,
            'bs'              : 10,
            'ms_channel'      : 1,
            'output_dir'      : None
        }
        
        nisqa_train = nisqaModel(nisqa_args_train)
                
        nisqa_args_test = {
            'mode'            : 'predict_dir',
            'pretrained_model': 'nisqa/weights/nisqa_tts.tar',
            'data_dir'        : 'report_files/wavs/test/',
            'num_workers'     : 0,
            'bs'              : 10,
            'ms_channel'      : 1,
            'output_dir'      : None
        }
        nisqa_test = nisqaModel(nisqa_args_test)

        #Exp.000   
        #MB:02.05.06 Measurements and Optimization Hyper Parameters
        GenerateRealWaveFilesForGeDis = True
        GenerateWaveFilesForGenerator = False
        GenerateWaveFilesForDiscrimin = True
        #####################################
        Metric_Measure_Active         = True
        print("\nMetric_Measure_Active is   :", Metric_Measure_Active)
        ######                 ##############
        Metric_Measure_MMCD           = False
        Metric_Measure_stoi           = False
        Metric_Measure_cosinesim      = True
        Metric_Measure_nisqa          = False

        Metric_Measure_IQAmetric      = True
        
        print("\nMetric_Measure_MMCD is     :", Metric_Measure_MMCD)
        print("Metric_Measure_stoi is     :", Metric_Measure_stoi)
        print("Metric_Measure_cosinesim is:", Metric_Measure_cosinesim)
        print("Metric_Measure_nisqa is    :", Metric_Measure_nisqa)
        Metric_Measure_stoi_net       = False
        Metric_Measure_pesq           = False
        Metric_Measure_WER            = False
        Metric_Measure_SDR            = False
        #####################################
        
        Metric_Optimize_Active        = True
        ######                       ########
        Metric_Optimize_MMCD          = False
        mmcdImpact    = 1.0 
        mmcdCycImpact = 1.0
    
        Metric_Optimize_stoi          = False
        stoiImpact    = 1.0
        stoiCycImpact = 1.0
        
        Metric_Optimize_cosinesim     = True
        cosineImpact    = 1.0
        cosineCycImpact = 1.0   
        
        Metric_Optimize_nisqa         = False
        nisqaImpact    = 1.0
        nisqaCycImpact = 1.0

        Metric_Optimize_IQAmetric     = False
        IQAImpact      = 1.0
        IQACycImpact   = 1.0
                    
        print("\nMetric_Optimize_Active is                :",Metric_Optimize_Active)
        print("\nMetric_Optimize_MMCD is                  :",Metric_Optimize_MMCD)
        print("mmcdImpact=",mmcdImpact, "mmcdCycImpact=",mmcdCycImpact)
        print("\nMetric_Optimize_stoi is                  :",Metric_Optimize_stoi)
        print("stoiImpact =",stoiImpact ,"stoiCycImpact=",stoiCycImpact)
        print("\nMetric_Optimize_cosinesim is             :",Metric_Optimize_cosinesim)
        print("cosineImpact =",cosineImpact ,"cosineCycImpact=",cosineCycImpact)
        print("\nMetric_Optimize_nisqa is                 :",Metric_Optimize_nisqa)
        print("nisqaImpact =",nisqaImpact ,"nisqaCycImpact=",nisqaCycImpact)
        Metric_Optimize_pesq          = False
        pesqImpact = 1.0
        Metric_Optimize_SDR           = False
        sdrImpact = 1.0
        #####################################            
        Metric_ShowPerEpoch_Active    = False
          
        Metric_Measure_Test_Active    = True  #Notion
        #####                        ########
        Metric_Measure_Test_mcd       = True   #MCD train Ref. RealA
        Metric_Measure_Test_stoi      = True
        Metric_Measure_Test_cosinesim = True
        Metric_Measure_Test_nisqa     = True
        
        print("Metric_Measure_Test_Active is   :",Metric_Measure_Test_Active)
        print("Metric_Measure_Test_mcd is      :",Metric_Measure_Test_mcd)
        print("Metric_Measure_Test_stoi is     :",Metric_Measure_Test_stoi)
        print("Metric_Measure_Test_cosinesim is:",Metric_Measure_Test_cosinesim)
        print("Metric_Measure_Test_nisqa is    :",Metric_Measure_Test_nisqa)
        
        Metric_Measure_Test_pesq      = False
        Metric_Measure_Test_SDR       = False
               
        Time_Align_TestFiles          = True
        
        EStoi_train = 'False'
        EStoi_test  = 'False'                
        
        vcc2018 = True
        CMU = False
        fs = 22050
        #fs = 16000
        sr = fs
        
        trainSRCspkEmbed = trainSF3spkEmbed
        trainTRGspkEmbed = trainTM1spkEmbed
        evalSRCspkEmbed  = evalSF3spkEmbed  
        evalTRGspkEmbed  = evalTM1spkEmbed
        
        #2
        # trainTRGspkEmbed = trainTF1spkEmbed
        # trainSRCspkEmbed = trainSF3spkEmbed 
        # evalSRCspkEmbed  = evalSF3spkEmbed 
        # evalTRGspkEmbed  = evalTF1spkEmbed
        
        #3
        # trainSRCspkEmbed = trainSM3spkEmbed 
        # trainTRGspkEmbed = trainTF1spkEmbed
        # evalSRCspkEmbed  = evalSM3spkEmbed 
        # evalTRGspkEmbed  = evalTF1spkEmbed
         
        #4    
        # trainSRCspkEmbed = trainSM3spkEmbed 
        # trainTRGspkEmbed = trainTM1spkEmbed
        # evalSRCspkEmbed  = evalSM3spkEmbed 
        # evalTRGspkEmbed  = evalTM1spkEmbed
        
        #print("\ntrainSRCspkEmbed.__str__()",trainSRCspkEmbed.__str__())
        #print("\ntrainSRCspkEmbed.__str__()",str(trainSRCspkEmbed))
                
        # print("trainTRGspkEmbed.__str__()")
        # print("evalSRCspkEmbed.__str__()" )
        # print("evalTRGspkEmbed.__str__()" )
        
        
        # print("\ntrainSRCspkEmbed",trainSRCspkEmbed,"trainTRGspkEmbed",trainTRGspkEmbed)
        # print("\nevalSRCspkEmbed" ,evalSRCspkEmbed,"evalTRGspkEmbed",evalTRGspkEmbed)
        
        stoiActivationEpoch       = 7000 
        PercepOptActivationEpoch1 = 5000
        PercepOptActivationEpoch2 = 7500
        
        tbTestLogPerEpo = 10
        
        damp_DFA        = 1.0
        damp_DFB        = 1.0
        damp_DCA        = 1.0 
        damp_DCB        = 1.0
        
        optmute         = 0.0
        extraLoss_fa    = 0
        extraLoss_fb    = 0
        extraLoss_ca    = 0
        extraLoss_cb    = 0
        
        print("PercepOptActivationEpoch1:",PercepOptActivationEpoch1)
        print("PercepOptActivationEpoch2:",PercepOptActivationEpoch2)
        
        print("\ntbTestLogPerEpo         :", tbTestLogPerEpo)
        print("damp_DFA=",damp_DFA,"   ; damp_DFB=",damp_DFB,"; damp_DCA=",damp_DCA,"; damp_DCB=",damp_DCB)        
        print("optmute=",optmute,"\nextraLoss_fa=",extraLoss_fa,"; extraLoss_fb=",extraLoss_fb)
        print("extraLoss_ca=",extraLoss_ca,"; extraLoss_cb=",extraLoss_cb)    
        
        ############################### Start of Training Epoches Loop #########################
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.logger.start_epoch()

            if self.logger.epoch  > stoiActivationEpoch:#MB:02.05.06
                Metric_Optimize_Active   = False
                Metric_Optimize_stoi     = False
                Metric_Optimize_MMCD     = False
                Metric_Optimize_cosinSim = False
                Metric_Optimize_nisqa    = False
                Metric_Optimize_SDR      = False
                
            if self.logger.epoch  > PercepOptActivationEpoch1:
                GenerateRealWaveFilesForGeDis = True
                GenerateWaveFsilesForDiscrimin = True
                
                Metric_Measure_Active    = True
                #etric_Measure_MMCD    = True
                Metric_Measure_cosinesim = True
                Metric_Measure_nisqa     = False
                
                Metric_Optimize_Active   = True
                Metric_Optimize_MMCD     = False
                Metric_Optimize_stoi     = False
                Metric_Optimize_MMCD     = False
                mmcdImpact    = 0.9 
                mmcdCycImpact = 0.6
                
                Metric_Optimize_cosinesim = True
                cosineImpact    = 1.0
                cosineCycImpact = 1.0   
                
                Metric_Optimize_nisqa    = False
                # nisqaImpact = 1.0
                # nisqaCycImpact = 0.1
                
                Metric_Optimize_SDR      = False
                
                #pass
                damp_DFA = 1.0
                damp_DFB = 1.0
                damp_DCA = 1.0 
                damp_DCB = 1.0
                
                optmute      = 0.0
                extraLoss_fa = 0.0
                extraLoss_fb = 0.0
                extraLoss_ca = 0.0
                extraLoss_cb = 0.0
                
                print("Metric_Optimize_Active                 :",Metric_Optimize_Active)
                print("Metric_Optimize_Active is true at Epoch:",self.logger.epoch)
                
                print("damp_DFA=",damp_DFA,"; damp_DFB=",damp_DFB,"; damp_DCA=",damp_DCA,"; damp_DCB=",damp_DCB)
                print("optmute=",optmute,"; extraLoss_fa=",extraLoss_fa,"; extraLoss_fb=",extraLoss_fb, "; extraLoss_ca=",extraLoss_ca,"; extraLoss_cb=",extraLoss_cb)   
                
            if self.logger.epoch  > PercepOptActivationEpoch2:
                Metric_Measure_Active    = True                
                Metric_Optimize_Active   = True
                Metric_Optimize_MMCD     = False
                Metric_Optimize_stoi     = False
                Metric_Measure_cosinesim = True
                Metric_Measure_nisqa     = False
                # mmcdImpact    = 0.9 
                # mmcdCycImpact = 0.6

                Metric_Optimize_cosinesim = True
                cosineImpact    = 1.0
                cosineCycImpact = 1.0                  
                
                Metric_Optimize_nisqa    = True
                # nisqaImpact = 1.0
                # nisqaCycImpact = 0.1
                
                
                Metric_Optimize_SDR      = False
                damp_DFA = 1.0
                damp_DFB = 1.0
                damp_DCA = 1.0 
                damp_DCB = 1.0
                optmute      = 1.0
                extraLoss_fa = 0.0
                extraLoss_fb = 0.0
                extraLoss_ca = 0.0
                extraLoss_cb = 0.0
                print("Metric_Optimize_Active                 :",Metric_Optimize_Active)
                print("Metric_Optimize_Active is true at Epoch:",self.logger.epoch)
                print("damp_DFA=",damp_DFA,"; damp_DFB=",damp_DFB,"; damp_DCA=",damp_DCA,"; damp_DCB=",damp_DCB)
                print("optmute=",optmute,"; extraLoss_fa=",extraLoss_fa,"; extraLoss_fb=",extraLoss_fb, "; extraLoss_ca=",extraLoss_ca,"; extraLoss_cb=",extraLoss_cb)   
                        
                
            ############### Start of Training Iterations Loop ###############
            for i, (real_A, mask_A, real_B, mask_B) in enumerate(tqdm(self.train_dataloader)):
                self.logger.start_iter()
                num_iterations = (
                    self.n_samples // self.mini_batch_size) * epoch + i
                    
                with torch.set_grad_enabled(True):
                    real_A = real_A.to(self.device, dtype=torch.float)
                    mask_A = mask_A.to(self.device, dtype=torch.float)
                    real_B = real_B.to(self.device, dtype=torch.float)
                    mask_B = mask_B.to(self.device, dtype=torch.float)
                    
                    
                    if (Metric_Optimize_Active and Metric_Optimize_stoi ):
                        pass
                    
                    # # m:03.02.17
                    # # ----------------
                    # # Train Discriminator
                    # # ----------------
                    # self.generator_A2B.eval()
                    # self.generator_B2A.eval()
                    # self.discriminator_A.train()
                    # self.discriminator_B.train()
                    # self.discriminator_A2.train()
                    # self.discriminator_B2.train()

                       
                    # #M:03.02.17 displacemen 
                    # # ----------------
                    # # Train Generator
                    # # ----------------
                    # self.generator_A2B.train()
                    # self.generator_B2A.train()
                    # self.discriminator_A.eval()
                    # self.discriminator_B.eval()
                    # self.discriminator_A2.eval()
                    # self.discriminator_B2.eval()

                    # # Generator Feed Forward
                    # fake_B = self.generator_A2B(real_A, mask_A)
                    # cycle_A = self.generator_B2A(fake_B, torch.ones_like(fake_B))
                    # fake_A = self.generator_B2A(real_B, mask_B)
                    # cycle_B = self.generator_A2B(fake_A, torch.ones_like(fake_A))
                    # identity_A = self.generator_B2A(
                    #     real_A, torch.ones_like(real_A))
                    # identity_B = self.generator_A2B(
                    #     real_B, torch.ones_like(real_B))
                    # d_fake_A = self.discriminator_A(fake_A)
                    # d_fake_B = self.discriminator_B(fake_B)
                    #
                    ########## Generating Audio Files #####################
                    #######################################################
                    #MB:02.05.03                   
                    fs=22050
                    realAfilePath ='report_files/wavs/train/real_A_wav020310.wav'
                    realBfilePath ='report_files/wavs/train/real_B_wav020310.wav' 
                    # fakeBfilePath ='report_files/wavs/train/fake_B_wav020430.wav'
                    # fakeAfilePath ='report_files/wavs/train/fake_A_wav020430.wav'
                    # cycgAfilePath ='report_files/wavs/train/cycg_A_wav020430.wav' 
                    # cycgBfilePath ='report_files/wavs/train/cycg_B_wav020430.wav' 
                    ###################### REAL Audio Files ########################################
                    if ( GenerateRealWaveFilesForGeDis ):
                        real_A_wav = decode_melspectrogram(self.vocoder,real_A[0].detach().cpu(),
                                                           self.dataset_A_mean, self.dataset_A_std ).cpu()   
                        #duration = len(real_A_wav)/ fs
                        #time = np.arange(0,duration,1/sample_rate) #time vector 
                        #print("\n real_A_wav duration:",duration)
                        torchaudio.save(realAfilePath, real_A_wav,sample_rate=fs)
                        real_A_wav = real_A_wav.squeeze()
                        real_B_wav = decode_melspectrogram(self.vocoder,real_B[0].detach().cpu(),
                                                           self.dataset_B_mean, self.dataset_B_std ).cpu()
                        #real_B_wav_int = (real_B_wav.numpy() * 32768).astype(np.int16)
                        torchaudio.save(realBfilePath, real_B_wav,sample_rate=fs)
                        real_B_wav = real_B_wav.squeeze()
                    ################ Fabricated Synthesized Files in Generators  ######################
                    ####################
                    # # Measurements in Generator
                    # if(Metric_Measure_Active and Metric_Measure_MMCD and False ): 
                    #     pass
                    # # Measurements in Generator
                    # if ( Metric_Measure_Active and Metric_Measure_stoi and False ):            
                    #     pass
                    # # Measurements in Generator
                    # if ( Metric_Measure_Active and Metric_Measure_cosinesim and False ): 
                    #     pass
                    # # Measurements in Generator
                    # if ( Metric_Measure_Active and Metric_Measure_nisqa and False ): 
                    #     pass                                                           
                    # ##################################################################################
                    # # For Two Step Adverserial Loss
                    # d_fake_cycle_A = self.discriminator_A2(cycle_A)
                    # d_fake_cycle_B = self.discriminator_B2(cycle_B)
                    # #
                    # # Generator Cycle Loss
                    # cycleLoss = torch.mean(
                    #     torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))
                    # #
                    # # Generator Identity Loss
                    # identityLoss = torch.mean(
                    #     torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))
                    # #
                    # # Generator Loss
                    # g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                    # g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)
                    # #
                    # # Generator Two Step Adverserial Loss
                    # generator_loss_A2B_2nd = torch.mean((1 - d_fake_cycle_B) ** 2)
                    # generator_loss_B2A_2nd = torch.mean((1 - d_fake_cycle_A) ** 2)
                    # #
                    # # if self.logger.epoch   <= 200: 
                    # #     alpha = 0.4
                    # #     betta = 0.8
                    # # elif self.logger.epoch <= 400:
                    # #     alpha = 0.4
                    # #     betta = 0.8
                    # # elif self.logger.epoch <= 600:
                    # #     alpha = 0.2
                    # #     betta = 0.8
                    # # elif self.logger.epoch  > 600:
                    # #     alpha = 0.2
                    # #     betta = 0.8
                    # # else:
                    # #     pass
                    # #
                    # # if self.logger.epoch   <= 200: 
                    # #     alpha = 1.0
                    # #     betta = 0.8
                    # #     kappa = 1.0
                    # # elif self.logger.epoch <= 400:
                    # #     alpha = 0.7
                    # #     betta = 0.8
                    # #     kappa = 1.0
                    # # elif self.logger.epoch <= 600:
                    # #     alpha = 0.4
                    # #     betta = 0.4
                    # #     kappa = 1.0
                    # # elif self.logger.epoch  > 600:
                    # #     alpha = 0.4
                    # #     betta = 0.8
                    # #     kappa = 1.0
                    # # else:
                    # #     pass
                    # alpha = 1.0
                    # betta = 1.0
                    # kappa = 1.0
                    # #
                    # # Total Generator Loss
                    # #MB:03.01.11
                    # g_loss = (g_loss_A2B + g_loss_B2A) + \
                    #     kappa*(generator_loss_A2B_2nd + generator_loss_B2A_2nd) + \
                    #     alpha * self.cycle_loss_lambda * cycleLoss + betta * self.identity_loss_lambda * identityLoss
                    #     #self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss
                    #    
                    # # Backprop for Generator
                    # self.reset_grad()
                    # g_loss.backward()
                    # self.generator_optimizer.step()

                    # ----------------------
                    # Train Discriminator
                    # ----------------------
                    self.generator_A2B.eval()
                    self.generator_B2A.eval()
                    self.discriminator_A.train()
                    self.discriminator_B.train()
                    self.discriminator_A2.train()
                    self.discriminator_B2.train()

                    # Discriminator Feed Forward
                    d_real_A = self.discriminator_A(real_A)
                    d_real_B = self.discriminator_B(real_B)
                    d_real_A2 = self.discriminator_A2(real_A)
                    d_real_B2 = self.discriminator_B2(real_B)
                    generated_A = self.generator_B2A(real_B, mask_B)
                    d_fake_A = self.discriminator_A(generated_A)

                    # For Two Step Adverserial Loss A->B
                    cycled_B = self.generator_A2B(
                        generated_A, torch.ones_like(generated_A))
                    d_cycled_B = self.discriminator_B2(cycled_B)

                    generated_B = self.generator_A2B(real_A, mask_A)
                    d_fake_B = self.discriminator_B(generated_B)

                    # For Two Step Adverserial Loss B->A
                    cycled_A = self.generator_B2A(
                        generated_B, torch.ones_like(generated_B))
                    d_cycled_A = self.discriminator_A2(cycled_A)

                    #MB:02.02.10  ##############################################################                                     
                    genBfilePath ='report_files/wavs/train/gen_B_wav020310.wav'
                    genAfilePath ='report_files/wavs/train/gen_A_wav020310.wav'
                    cycBfilePath ='report_files/wavs/train/cyc_B_wav020310.wav'
                    cycAfilePath ='report_files/wavs/train/cyc_A_wav020310.wav'                 
                    
                    # Measurements in Training of Discriminator
                    if ( GenerateWaveFilesForDiscrimin ):
                        # gen_A_wav = decode_melspectrogram(self.vocoder,generated_A[0].detach().cpu(), self.dataset_B_mean,
                        #                                   self.dataset_B_std).cpu()
                        gen_A_wav = decode_melspectrogram(self.vocoder,generated_A[0].detach().cpu(), self.dataset_A_mean,
                                                          self.dataset_A_std).cpu()
                        torchaudio.save(genAfilePath, gen_A_wav, sample_rate=fs)
                        gen_A_wav = gen_A_wav.squeeze() 

                        # gen_B_wav = decode_melspectrogram(self.vocoder,generated_B[0].detach().cpu(), self.dataset_A_mean,
                        #                                   self.dataset_A_std).cpu()
                        gen_B_wav = decode_melspectrogram(self.vocoder,generated_B[0].detach().cpu(), self.dataset_B_mean,
                                                          self.dataset_B_std).cpu()
                        torchaudio.save(genBfilePath, gen_B_wav, sample_rate=fs)
                        #MB:02.03.08
                        gen_B_wav = gen_B_wav.squeeze() 
                        
                        cyc_A_wav = decode_melspectrogram(self.vocoder,cycled_A[0].detach().cpu(), self.dataset_A_mean,
                                                          self.dataset_A_std).cpu()  
                        torchaudio.save(cycAfilePath, cyc_A_wav, sample_rate=fs)
                        cyc_A_wav = cyc_A_wav.squeeze()

                        cyc_B_wav = decode_melspectrogram(self.vocoder, cycled_B[0].detach().cpu(), self.dataset_B_mean, 
                                                          self.dataset_B_std).cpu()
                        torchaudio.save(cycBfilePath, cyc_B_wav, sample_rate=fs)                
                        cyc_B_wav = cyc_B_wav.squeeze()

                    # #####################################################################
                    
                    # Measurements in Discriminator Training
                    if( Metric_Measure_Active and Metric_Measure_MMCD ): 
                        #print("\nMetric_Measure_Active & Metric_Measure_MMCD:True")
                        #print(f'\n shape: {real_A_wav}')
                        
                        mcd_value1 = mcd_toolbox_train.calculate_mcd(real_A_wav.numpy(),
                                                                     gen_B_wav.numpy())
                        #print("\nmcd_value1",mcd_value1)
                        mcd_value3 = mcd_toolbox_train.calculate_mcd(real_B_wav.numpy(),
                                                                      cyc_B_wav.numpy())   
                        #print("\nmcd_value3",mcd_value3)
                        
                        #MMCD_GBTA =  mcd_value1
                        MMCD_GBTA = 1.0 - ( mcd_value1 / maxpymcd_train )
                        #MMCD_GBTA = ( mcd_value1 / maxpymcd_train )
                        
                        MMCD_CBTB = 1.0 - ( mcd_value3 / maxpymcd_train ) 
                                              
                        # MMCD_CBTB =  mcd_value3 / maxpymcd_train  
                        # MMCD_CATA =  mcd_value4 / maxpymcd_train 
                        
                        if Metric_Optimize_Active and  Metric_Optimize_MMCD :
                            #print("Metric_Optimize_MMCD is Active")
                            mcd_value2 = mcd_toolbox_train.calculate_mcd(real_B_wav.numpy(),
                                                                gen_A_wav.numpy())
                            mcd_value4 = mcd_toolbox_train.calculate_mcd(real_A_wav.numpy(),
                                                               cyc_A_wav.numpy())
                            MMCD_GATB = 1.0 - ( mcd_value2 / maxpymcd_train )
                            #MMCD_GATB =  ( mcd_value2 / maxpymcd_train )
                            MMCD_CATA = 1.0 - ( mcd_value4 / maxpymcd_train ) 
                        
                        
                        ##################################################################
                        # mcd_value5 = mcd_toolbox_train.calculate_mcd(real_B_wav.numpy(),
                        #                                         gen_B_wav.numpy())
                        # mcd_value6 = mcd_toolbox_train.calculate_mcd(real_A_wav.numpy(),
                        #                                         gen_A_wav.numpy())
                        
                                           
                        # import math
                        # MMCD_GBTA = 1.0 -  math.exp(-mcd_value1) 
                        # MMCD_GATB = 1.0 -   math.exp(-mcd_value2)
                        # MMCD_CBTB = 10.0 / abs(10.0 + mcd_value3)  
                        # MMCD_CATA = 10.0 / abs(10.0 + mcd_value4)   
                        
                        # MMCD_GBTA = 10.0 / abs(10.0 + mcd_value1) 
                        # MMCD_GATB = 10.0 / abs(10.0 + mcd_value2)  
                        # MMCD_CBTB = 10.0 / abs(10.0 + mcd_value3)  
                        # MMCD_CATA = 10.0 / abs(10.0 + mcd_value4)   
                        
                        # import math
                        # def sigmoid(x):
                        #     return 1- 1 / (1 + 0.01*math.exp(-x))
                        
                        # MMCD_FTB = sigmoid(mcd_value1)  # Variable between 0 to 1
                        # MMCD_FTB = sigmoid(mcd_value2)
                        # MMCD_CTB = sigmoid(mcd_value3)
                        # MMCD_CTA = sigmoid(mcd_value4)
                        
                        # print("\n*** MMCD_GBTA",MMCD_GBTA)
                        # print("\n*** MMCD_GATB",MMCD_GATB)  
                        # print("\n*** MMCD_CBTB",MMCD_CBTB)
                        # print("\n*** MMCD_CATA",MMCD_CATA)  # TODO delete
                        
                        with open('report_files/MMCD_GBTA.txt', 'a') as f01:
                            f01.write(str( MMCD_GBTA )+"\n")
                        
                        # self.logger._log_scalars( scalar_dict={'A.3_1.Train_mmcd_gbta': MMCD_GBTA,
                        #                                        'A.3_2.Train_mmcd_gatb': MMCD_GATB, 
                        #                                        'A.3_3.Train_mmcd_cbtb': MMCD_CBTB,
                        #                                        'A.3_4.Train_mmcd_cata': MMCD_CATA} ,
                        #                          print_to_stdout = False )
                     
                    
                    
                        self.logger._log_scalars( scalar_dict={'A.3_1.Train_mmcd_gbta': MMCD_GBTA,
                                                               'A.3_3.Train_mmcd_cbtb': MMCD_CBTB} ,
                                                     print_to_stdout = False )

                    
                    # Measurements in Discriminator
                    if ( Metric_Measure_Active and Metric_Measure_stoi ):
                        #print("\nMetric_Measure_active & Metric_Measure_stoi:True")
                        #print("Measuring STIO score by pystoi is activated")                       
                        
                        #EStoi_train='True'
                        # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
                        #gen_B_wav2 , fs = torchaudio.load(genBfilePath)
                        #real_A_wav2 , fs = torchaudio.load(realAfilePath)
                        #realA_genB_stoi = stoi(real_A_wav2.numpy().(),\
                          #                      gen_B_wav2.numpy().squeeze(),fs,extended=False)
                        #realA_genB_stoi = stoi(real_A_wav1,gen_B_wav1,fs,extended=False)
                        """
                        #MB:02.05.09
                        import torch
                        from torch import nn
                        from torch_stoi import NegSTOILoss

                        sample_rate = 16000
                        loss_func = NegSTOILoss(sample_rate=sample_rate)
                        # Your nnet and optimizer definition here
                        nnet = nn.Module()

                        noisy_speech = torch.randn(2, 16000)
                        clean_speech = torch.randn(2, 16000)
                        # Estimate clean speech
                        est_speech = nnet(noisy_speech)
                        # Compute loss and backward (then step etc...)
                        loss_batch = loss_func(est_speech, clean_speech)
                        loss_batch.mean().backward()
                        """
                        # realA_genB_stoi = stoi(real_A_wav.numpy(),
                        #                         gen_B_wav.numpy(),fs,extended=EStoi)
                        # realB_genA_stoi = stoi(real_B_wav.numpy(),
                        #                         gen_A_wav.numpy(),fs,extended=EStoi)
                        # realB_cycB_stoi = stoi(real_B_wav.numpy(),
                        #                         cyc_B_wav.numpy(),fs,extended=EStoi)
                        # realA_cycA_stoi = stoi(real_A_wav.numpy(),
                        #                         cyc_A_wav.numpy(),fs,extended=EStoi)
                        
                        '''
                        realA_genB_stoi = stoi(real_A_wav.numpy(),
                                                 gen_B_wav.numpy(),fs,extended=EStoi_train)
                        # realB_genA_stoi = stoi(real_B_wav.numpy(),
                        #                          gen_A_wav.numpy(),fs,extended=EStoi_train)
                        realB_cycB_stoi = stoi(real_B_wav.numpy(),
                                                cyc_B_wav.numpy(),fs,extended=EStoi_train)
                        # realA_cycA_stoi = stoi(real_A_wav.numpy(),
                        #                         cyc_A_wav.numpy(),fs,extended=EStoi_train)
                        if Metric_Optimize_Active  and  Metric_Optimize_stoi :
                            realB_genA_stoi = stoi(real_B_wav.numpy(),
                                                  gen_A_wav.numpy(),fs,extended=EStoi_train)
                            realA_cycA_stoi = stoi(real_A_wav.numpy(),
                                                  cyc_A_wav.numpy(),fs,extended=EStoi_train)
                        '''
        
                        # MB:02.11.24:
                        #from torchmetrics.audio import ShortTimeObjectiveIntelligibility 
                        stoi_torch = ShortTimeObjectiveIntelligibility(22050, False)
                        
                        floor_stoi = 0.5
                        realA_genB_stoi = stoi_torch(gen_B_wav ,real_A_wav)
                        if realA_genB_stoi > 0.1 : floor_stoi = realA_genB_stoi
                        realA_genB_stoi = floor_stoi
                        
                        floor_stoi = 0.6
                        realB_cycB_stoi = stoi_torch(cyc_B_wav, real_B_wav )
                        if realB_cycB_stoi > 0.2 : floor_stoi = realB_cycB_stoi
                        realB_cycB_stoi = floor_stoi
                        
                        if Metric_Optimize_Active  and  Metric_Optimize_stoi : 
                            floor_stoi = 0.5
                            realB_genA_stoi = stoi_torch( gen_A_wav,real_B_wav)
                            if realB_genA_stoi > 0.1 : floor_stoi = realB_genA_stoi
                            realB_genA_stoi = floor_stoi

                            floor_stoi = 0.6
                            realA_cycA_stoi = stoi_torch(  cyc_A_wav,real_A_wav)
                            if realA_cycA_stoi > 0.2 : floor_stoi = realA_cycA_stoi
                            realA_cycA_stoi = floor_stoi
                            
                        # print("realA_genB_stoi",realA_genB_stoi)
                        # #print("realB_genA_stoi",realB_genA_stoi)
                        # print("realB_cycB_stoi",realB_cycB_stoi)
                        # #print("realA_cycA_stoi",realA_cycA_stoi) # TODO delete
                        
                        with open("report_files/realA_genB_stoi.txt", "a") as f02:
                            f02.write(str(realA_genB_stoi)+"\n")
                        
                        # self.logger._log_scalars( scalar_dict={'A.4_1.Train_stoi_realAgenB': realA_genB_stoi,
                        #                                        'A.4_2.Train_stoi_realBgenA': realB_genA_stoi,
                        #                                        'A.4_3.Train_stoi_realBcycB': realB_cycB_stoi,
                        #                                        'A.4_4.Train_stoi_realAcycA': realA_cycA_stoi},
                        #                          print_to_stdout = False )
                        
                        self.logger._log_scalars( scalar_dict={'A.4_1.Train_stoi_realAgenB': realA_genB_stoi,
                                                               'A.4_3.Train_stoi_realBcycB': realB_cycB_stoi},
                                                 print_to_stdout = False )
                        
                        
                        # file02 = open('report_files/realA_genB_stoi.txt', 'a')# append mode     
                        # file02.write(str(realA_genB_stoi )+"\n")
                        # file02.close() 
                        
                    # Measurements in Discriminator
                    if ( Metric_Measure_Active and Metric_Measure_cosinesim ):
                        #print("\nMetric_Measure_Active & Metric_Measure_cosinesim: True")
                        # sr = 22050
                        # #sr=16000
                        # print("\n\ngen_B_wav.shape",gen_B_wav.shape)                       
                        # #mel_genb = model_gru.melspec_from_file( genBfilePath).cuda()
                        # mel_genb = model_gru.melspec_from_array( gen_B_wav , sr )
            
                        #print("\n\n gen_B_wav.shape:", gen_B_wav.shape )      
                            
                        if ( vcc2018 == True ): 
                            fs=22050
                            sr = 22050
                            # mel_genb = model_gru.melspec_from_array( gen_B_wav , sr )
                            # mel_gena = model_gru.melspec_from_array( gen_A_wav , sr )
                            # mel_reab = model_gru.melspec_from_array( real_B_wav, sr )
                            # mel_reaa = model_gru.melspec_from_array( real_A_wav, sr )
                            # mel_cycb = model_gru.melspec_from_array( cyc_B_wav , sr )
                            # mel_cyca = model_gru.melspec_from_array( cyc_A_wav , sr )
            
                            #model_gru.cuda()
                
                            #print("mel1.type ",mel1.type  )
                            #print("mel1.size() ",mel1.size()  )
                            #embed1 = model(mel1[None].cuda().requires_grad_(True))
                            #include [None] to add the batch dimension
                            
                            # embed_genb = model_gru(mel_genb[None].cuda() ).cuda() #GenB
                            # embed_gena = model_gru(mel_gena[None].cuda() ).cuda() #GenA
                            # embed_reab = model_gru(mel_reab[None].cuda() ).cuda() #RealB
                            # embed_reaa = model_gru(mel_reaa[None].cuda() ).cuda()# RealA, include [None] to add the batch dimension
                            #MB:02.03.03 adding for cycled wav
                            # embed_cycb = model_gru(mel_cycb[None].cuda() ).cuda()
                            # embed_cyca = model_gru(mel_cyca[None].cuda() ).cuda()

                            #cos = torch.nn.CosineSimilarity(dim=1)                            
                            #cosinesim_realBbetta=cos(embed_reab, betta_norm )
                            #print('\ncosinesim_realBbetta',cosinesim_realBbetta)
                            
                            # cosinesim_genBtarg=cos(embed_genb, embed_reab )
                            # cosinesim_genAtarg=cos(embed_gena, embed_reaa ) 
                            # cosinesim_cycBtarg=cos(embed_cycb, embed_reab )
                            # cosinesim_cycAtarg=cos(embed_cyca, embed_reaa )
                            
                            #cosinesim_genBtarg=cos(embed_genb, betta_norm )                            
                            #cosinesim_genAtarg=cos(embed_gena, alpha_norm )               
                            #cosinesim_genBsorc=cos(embed_genb, alpha_norm ) 
                            # Converted B Should have far distance from source alpha more and more, so can be used negetively in LossFuncOpt.
                            #cosinesim_genAsorc=cos(embed_gena, betta_norm )       
                            #cosinesim_cycBtarg=cos(embed_cycb, betta_norm )             
                            #cosinesim_cycAtarg=cos(embed_cyca, alpha_norm )         
                            
                            #Based on Resemblyzer
                            # fpath = Path("path_to_an_audio_file")
                            # wav = preprocess_wav(fpath)
                            #encoder = VoiceEncoder()
                            #resemb_encoder = VoiceEncoder()
                            # embed = encoder.embed_utterance(wav)
                            # np.set_printoptions(precision=3, suppress=True)
                            # print(embed)
                            
                            wav_realB  = preprocess_wav(realBfilePath)
                            #wav_realA  = preprocess_wav(realAfilePath)
                            wav_genB   = preprocess_wav(genBfilePath )
                            #wav_genA   = preprocess_wav(genAfilePath )
                            wav_cycB   = preprocess_wav(cycBfilePath )
                            #wav_cycA   = preprocess_wav(cycAfilePath )
                            
                            embed_realB = torch.from_numpy(resemb_encoder.embed_utterance(wav_realB)).cuda()
                            #embed_realA = torch.from_numpy(resemb_encoder.embed_utterance(wav_realA))
                            embed_genB  = torch.from_numpy(resemb_encoder.embed_utterance(wav_genB )).cuda()
                            #embed_genA  = torch.from_numpy(resemb_encoder.embed_utterance(wav_genA ))
                            embed_cycB  = torch.from_numpy(resemb_encoder.embed_utterance(wav_cycB )).cuda()
                            #embed_cycA  = torch.from_numpy(resemb_encoder.embed_utterance(wav_cycA ))
                            # # np.set_printoptions(precision=3, suppress=True)
                            # # print(embed)
                            
                            cos = torch.nn.CosineSimilarity(dim=0)
                            #MB:02.08.13
                            # trainSF3spkEmbed #As a Representative of Training Source Speaker by Resemblyzer
                            # trainTM1spkEmbed #As a Representative of Training Target Speaker
                            # evalSF3spkEmbed  #As a Representative of Source Evaluation Speaker
                            # evalTM1spkEmbed  #As a Representative of Target Evaluation(Reference) Speaker:
                            #torch.load( 'report_files/evalTM1spkEmbed.pt').cuda()
                            
                            cosinesim_genBtarg=cos(embed_genB, embed_realB )
                            #cosinesim_genBtarg=cos(embed_genB, trainTM1spkEmbed )
                            #cosinesim_genBtarg=cos(embed_genB, trainTRGspkEmbed )
                            #cosinesim_genAtarg=cos(embed_genA, embed_realA ) 
                            #cosinesim_genAtarg=cos(embed_genA, trainSF3spkEmbed )
                            cosinesim_cycBtarg= cos(embed_cycB, embed_realB )
                            #cosinesim_cycBtarg= cos(embed_cycB, trainTRGspkEmbed )
                            #cosinesim_cycAtarg= cos(embed_cycA, embed_realA )
                            #cosinesim_cycAtarg= cos(embed_cycA, trainSF3spkEmbed )
                            
                            #MB:03.01.20 Loss for SpeakerEmbeddingSimilarityDistance(SESD)
                            # Lsesd_genB = torch.sqrt(torch.sum((embed_genB,- embed_realB)**2)  )
                            # Lsesd_genA = torch.sqrt(torch.sum((embed_genA,- embed_realA)**2)  )
                            # Lsesd_cycB = torch.sqrt(torch.sum((embed_cycB,- embed_realB)**2)  )
                            # Lsesd_cycA = torch.sqrt(torch.sum((embed_cycA,- embed_realA)**2)  )   
                                                
                            
                            if Metric_Optimize_Active  and  Metric_Optimize_cosinesim :
                                wav_realA  = preprocess_wav(realAfilePath)
                                wav_genA   = preprocess_wav(genAfilePath )
                                wav_cycA   = preprocess_wav(cycAfilePath )
    
                                embed_realA = torch.from_numpy(resemb_encoder.embed_utterance(wav_realA)).cuda()
                                embed_genA  = torch.from_numpy(resemb_encoder.embed_utterance(wav_genA )).cuda()
                                embed_cycA  = torch.from_numpy(resemb_encoder.embed_utterance(wav_cycA )).cuda()
                                
                                cosinesim_genAtarg=cos(embed_genA, embed_realA ) 
                                #cosinesim_genAtarg=cos(embed_genA, trainSRCspkEmbed )
                                cosinesim_cycAtarg= cos(embed_cycA, embed_realA )
                                #cosinesim_cycAtarg= cos(embed_cycA, trainSRCspkEmbed )
                                
                                #MB:03.01.20 Loss for SpeakerEmbeddingSimilarityDistance(SESD)
                                # Lsesd_genB = torch.sqrt(torch.sum((embed_genB - embed_realB)**2)  )
                                # Lsesd_genA = torch.sqrt(torch.sum((embed_genA - embed_realA)**2)  )
                                # Lsesd_cycB = torch.sqrt(torch.sum((embed_cycB - embed_realB)**2)  )
                                # Lsesd_cycA = torch.sqrt(torch.sum((embed_cycA - embed_realA)**2)  )
                                #

                                #MB:03.01.21-1070
                                # a=(embed_genB - embed_realB)**2
                                # b = torch.sum((embed_genB - embed_realB)**2)
                                # print(f'\n(embed_genB - embed_realB)**2).size()= {a.size() }') 
                                # print(f'\ntorch.sum((embed_genB - embed_realB)**2).size()= {b.size()}')
                                # print(f'\nLsesd_cycB: {Lsesd_cycB}') 
                                # print(f'\nLsesd_genB: {Lsesd_genB}') 
                                # print(f'\nLsesd_cycA: {Lsesd_cycA}') 
                                # print(f'\nLsesd_genA: {Lsesd_genA}') 

                                #MB:03.01.22 Loss for SpeakerEmbeddingSimilarityMetric(SESM)
                                # SESMetric_genB = 1 - torch.sqrt(torch.sum((embed_genB - embed_realB)**2)  )
                                # SESMetric_genA = 1 - torch.sqrt(torch.sum((embed_genA - embed_realA)**2)  )
                                # SESMetric_cycB = 1 - torch.sqrt(torch.sum((embed_cycB - embed_realB)**2)  )
                                # SESMetric_cycA = 1 - torch.sqrt(torch.sum((embed_cycA - embed_realA)**2)  )
        
                            
                        if ( CMU == True ): 
                            fs = 16000
                            sr = 16000
                            # model_gru.cuda()
                            # mel_genb = model_gru.melspec_from_file( genBfilePath).cuda()
                            # mel_gena = model_gru.melspec_from_file( genAfilePath).cuda()
                            # mel_reab = model_gru.melspec_from_file(realBfilePath).cuda()
                            # mel_reaa = model_gru.melspec_from_file(realAfilePath).cuda()
                            # mel_cycb = model_gru.melspec_from_file( cycBfilePath).cuda()
                            # mel_cyca = model_gru.melspec_from_file( cycAfilePath).cuda()
                            
                            # if you were using the `convgru_embedder`, you can go:
                            # import librosa
                            # wav, _ = librosa.load('example.wav', sr=16000)
                            # wav = torch.from_numpy(wav).float()
                            
                            model_convgru.cuda()
                            #embedding = model_convgru(wav[None])
                            embed_genb = model_convgru( gen_B_wav[None].cuda() ).cuda()
                            #embed_gena = model_convgru( gen_A_wav[None].cuda() ).cuda()
                            embed_reab = model_convgru(real_B_wav[None].cuda() ).cuda()
                            #embed_reaa = model_convgru(real_A_wav[None].cuda() ).cuda()
                            embed_cycb = model_convgru( cyc_B_wav[None].cuda() ).cuda()
                            #embed_cyca = model_convgru( cyc_A_wav[None].cuda() ).cuda()
                        
                            cos = torch.nn.CosineSimilarity(dim=1)
                            #cosinesim_realBbetta=cos(embed_reab, delta_norm )
                            #print('\ncosinesim_realBdelta',cosinesim_realBdelta )
                            
                            cosinesim_genBtarg=cos(embed_genb, embed_reab )
                            #cosinesim_genBtarg=cos(embed_genb, delta_norm )
                            #  cosinesim_genAtarg=cos(embed_gena, embed_reaa )
                            #cosinesim_genAtarg=cos(embed_gena, kapa_norm )               
                                    
                            #cosinesim_cycBtarg=cos(embed_cycb, betta_norm )             
                            cosinesim_cycBtarg=cos(embed_cycb, embed_reab )
                            #cosinesim_cycAtarg=cos(embed_cyca, alpha_norm )         
                            #  cosinesim_cycAtarg=cos(embed_cyca, embed_reaa )

                            #cosinesim_cycBsorc=cos(embed_cycb, delta_norm )         
                            #cosinesim_cycAsorc=cos(embed_cyca, kapa_norm )
                        
                        ################################################
                        
                        # print("\ncosinesim_genBtarg",cosinesim_genBtarg)
                        # #print("\ncosinesim_genAtarg",cosinesim_genAtarg)
                        # print("\ncosinesim_cycBtarg",cosinesim_cycBtarg)
                        # #print("\ncosinesim_cycAtarg",cosinesim_cycAtarg)
                        
                        # print("\ncosinesim_genBsorc",cosinesim_genBsorc)
                        # print("\ncosinesim_genAsorc",cosinesim_genAsorc)
                        
                        # file03 = open("report_files/cosinesim_genBtarg.txt", "a")# append mode     
                        # file03.write(str(cosinesim_genBtarg.detach().cpu())+"\n")   # TODO delete
                        # file03.close()
                        with open("report_files/cosinesim_genBtarg.txt", "a") as f03:
                            f03.write(str(cosinesim_genBtarg.detach().cpu())+"\n")
                        
                        # self.logger._log_scalars( scalar_dict={'A.5_1.Train_CosineSim_genBtarg':\
                        #                                        cosinesim_genBtarg.detach().cpu().numpy() ,
                        #                                        'A.5_2.Train_CosineSim_genAtarg':\
                        #                                        cosinesim_genAtarg.detach().cpu().numpy() ,
                        #                                        'A.5_3.Train_CosineSim_cycBtarg':\
                        #                                        cosinesim_cycBtarg.detach().cpu().numpy() ,
                        #                                        'A.5_4.Train_CosineSim_cycAtarg':\
                        #                                        cosinesim_cycAtarg.detach().cpu().numpy() },
                        #                          print_to_stdout = False)
                        
                        self.logger._log_scalars( scalar_dict={'A.5_1.Train_CosineSim_genBtarg':\
                                                               cosinesim_genBtarg.detach().cpu().numpy() ,
                                                               'A.5_3.Train_CosineSim_cycBtarg':\
                                                               cosinesim_cycBtarg.detach().cpu().numpy()},
                                                 print_to_stdout = False)
                
                
                
                    # Measurements in Training Discriminator    
                    if ( Metric_Measure_Active and Metric_Measure_nisqa ): 
                        #print("\nMetric_Measure_Active & Metric_Measure_nisqa: True")
                        # path = "report_files/wavs/train/"

                        # Nisqatts_RB_norm = run_predict.nisqa_score(path + 'real_B_sound020310.wav')/5.0
                        # Nisqatts_RA_norm = run_predict.nisqa_score(path + 'real_A_sound020310.wav')/5.0
                        # Nisqatts_GB_norm = run_predict.nisqa_score(path +  'gen_B_sound020430.wav')/5.0
                        # Nisqatts_GA_norm = run_predict.nisqa_score(path +  'gen_A_sound020430.wav')/5.0                                   
                        # Nisqatts_CB_norm = run_predict.nisqa_score(path +  'cyc_B_sound020430.wav')/5.0     
                        # Nisqatts_CA_norm = run_predict.nisqa_score(path +  'cyc_A_sound020430.wav')/5.0
                        
                        nisqa_train._loadDatasets()
                        mos_train = nisqa_train.predict().to_dict()
                        
                        mos_train = dict( zip(mos_train['deg'].values(), mos_train['mos_pred'].values()) )

                        Nisqatts_GB_norm = mos_train[ 'gen_B_wav020310.wav'] / 5.0
                        #Nisqatts_GA_norm = mos_train[ 'gen_A_wav020310.wav'] / 5.0
                        Nisqatts_CB_norm = mos_train[ 'cyc_B_wav020310.wav'] / 5.0
                        #Nisqatts_CA_norm = mos_train[ 'cyc_A_wav020310.wav'] / 5.0
                       
                        # Nisqatts_RB_norm = mos_train['real_B_wav020310.wav'] / 5.0
                        # Nisqatts_RA_norm = mos_train['real_A_wav020310.wav'] / 5.0
                    
                         
                        #Nisqatts_CA_norm = run_predict.nisqa_score(path + cycAfilePath )/ 5.0 
                        #cycAfilePath='report_files/wavs/cyc_A_sound020430.wav'
                        
                        if Metric_Optimize_nisqa :
                            Nisqatts_GA_norm = mos_train[ 'gen_A_wav020310.wav'] / 5.0
                            Nisqatts_CA_norm = mos_train[ 'cyc_A_wav020310.wav'] / 5.0
                                                    # print("\nNisqatts_GB_norm",Nisqatts_GB_norm)
                        # print("\nNisqatts_GA_norm",Nisqatts_GA_norm)
                        # print("\nNisqatts_RB_norm",Nisqatts_RB_norm)
                        # print("\nNisqatts_RA_norm",Nisqatts_RA_norm) # TODO delete
                        
                        
                        
                        # file04= open('report_files/Nisqatts_GB_norm.txt', 'a')# append mode     
                        # file04.write(str(Nisqatts_GB_norm )+"\n")  # TODO delete
                        # file04.close()
                        with open("report_files/Nisqatts_GB_norm.txt", 'a') as f04:
                            f04.write( str(Nisqatts_GB_norm ) + "\n")
            
                        # self.logger._log_scalars( scalar_dict={'A.6_1.Train_Nisqatts_GBnorm': Nisqatts_GB_norm,
                        #                                        'A.6_2.Train_Nisqatts_GAnorm': Nisqatts_GA_norm,
                        #                                        'A.6_3.Train_Nisqatts_CBnorm': Nisqatts_CB_norm,
                        #                                        'A.6_4.Train_Nisqatts_CAnorm': Nisqatts_CA_norm},
                        #                          print_to_stdout = False)
                        
                        self.logger._log_scalars( scalar_dict={'A.6_1.Train_Nisqatts_GBnorm': Nisqatts_GB_norm,
                                                               'A.6_3.Train_Nisqatts_CBnorm': Nisqatts_CB_norm},
                                                 print_to_stdout = False)
                        
                        # self.logger._log_scalars( scalar_dict={'A.6_1.Train_Nisqatts_GBnorm': Nisqatts_GB_norm},
                        #                          print_to_stdout = False)
                        
                    
                    if (Metric_Measure_Active and Metric_Measure_IQAmetric  ):
                        #print(f'\nMetric_Measure_IQAmetric:{Metric_Measure_IQAmetric}')
                        cycled_A    =    cycled_A.unsqueeze(0)
                        real_A      =      real_A.unsqueeze(0)
                        cycled_B    =    cycled_B.unsqueeze(0)
                        real_B      =      real_B.unsqueeze(0)
                        generated_A = generated_A.unsqueeze(0)
                        generated_B = generated_B.unsqueeze(0)
                        #
                        IQAmetric1 = FSIM(channels=1).to(self.device)
                        IQAmetric2 = SSIM(channels=1).to(self.device)
                        IQAmetric3 = GMSD(channels=1).to(self.device)
                        #
                        #metric4 = LPIPSvgg(channels=1).to(self.device)
                        #metric5 = VIF(channels=1).to(self.device)
                        #metric6 = MAD(channels=1).to(self.device)
                        #metric7 = NLPD(channels=1).to(self.device)
                        #
                        # cycled_A = Variable(cycled_A.float().to(self.device), requires_grad=True)
                        # real_A   = Variable(  real_A.float().to(self.device), requires_grad=True)
                        # cycled_B = Variable(cycled_B.float().to(self.device), requires_grad=True)
                        # real_B   = Variable(  real_B.float().to(self.device), requires_grad=True)
                        #
                        # print('\n cycle_A.shape and type(cycle_A)= ',cycle_A.shape, type(cycle_A))
                        # print('\n real_A.shape and type(real_A)= ',real_A.shape,type(real_A))
                        #lr = 0.005
                        #lr = self.discriminator_lr
                        #metric1_optimizer = torch.optim.Adam([cycle_A], lr=lr)
                        #metric1_optimizer = torch.optim.Adam([cycle_A], lr=lr)
                        #
                        # metric1distA = metric1(cycled_A, real_A,as_loss=True)
                        # metric1distB = metric1(cycled_B, real_B,as_loss=True)
                        # metric1dist=(metric1distA + metric1distB) /2
                        #print('\n mrh metric1 iter',metric1dist.item())

                        IQAmetric1genA = IQAmetric1(generated_A, real_A , as_loss=False).detach()
                        IQAmetric1genB = IQAmetric1(generated_B, real_B , as_loss=False).detach()
                        IQAmetric1cycA = IQAmetric1(cycled_A   , real_A , as_loss=False).detach()
                        IQAmetric1cycB = IQAmetric1(cycled_B   , real_B , as_loss=False).detach()
                        #
                        #IQAmetric1cyc  = 10 * ( IQAmetric1cycA + IQAmetric1cycB ) / 2.0
                        # dist2A = metric2(cycled_A, real_A)
                        # dist2B = metric2(cycled_B, real_B)
                        # dist2  = 10 * (dist2A +dist2B) /2
                        # dist3A = metric3(cycled_A, real_A)
                        # dist3B = metric3(cycled_B, real_B)
                        # dist3  = 10 * (dist3A +dist3B) /2
                        # print('iter: %d, IQAmetric1CycB: %.3g' % (i, IQAmetric1cycB.item()))  
                        # print('iter: %d, IQAmetric1cycA: %.3g' % (i, IQAmetric1cycA.item()))
                        #
                        cycled_A = cycled_A.squeeze(0)
                        cycled_B = cycled_B.squeeze(0)
                        real_A   =   real_A.squeeze(0)
                        real_B   =   real_B.squeeze(0)          
                        #                        
                        self.logger._log_scalars( scalar_dict={'A.15_1.Train_IQAmetric1cycB':\
                                                               IQAmetric1cycB ,
                                                               'A.15_2.Train_IQAmetric1cycA':\
                                                               IQAmetric1cycA },
                                                 print_to_stdout = False )

                    
                    #MB:02.05.22 Assessment MaskOut files in training  by PESQ
                    # Measurements in Training Discriminator    
                    if ( Metric_Measure_Active and Metric_Measure_pesq ): 
                        #print("\nMetric_Measure_Active & Metric_Measure_pesq: True")
                        # path = "report_files/wavs/train/"

                        # Run on 1080
                        #import soundfile as sf
                        #from pypesq import pesq

                        # realAfile ='report_files/wavs/train/real_A_sound020310.wav'
                        # realBfile ='report_files/wavs/train/real_B_sound020310.wav' 
                        # genAfile  ='report_files/wavs/train/gen_A_sound020310.wav' 
                        # genBfile  ='report_files/wavs/train/gen_B_sound020310.wav' 
                        # cycAfile  ='report_files/wavs/train/cyc_A_sound020310.wav' 
                        # cycBfile  ='report_files/wavs/train/cyc_B_sound020310.wav' 

                        # realB, fs = sf.read(realBfile)
                        # realA, fs = sf.read(realAfile)
                        # genB , fs = sf.read(genBfile)
                        # genA , fs = sf.read(genAfile)
                        # cycB , fs = sf.read(cycBfile)
                        # cycA , fs = sf.read(cycAfile)

                        # Clean and den should have the same length, and be 1D

                        realA_genB_tr_pesqnorm = ( pesq( real_A_wav , gen_B_wav ) + 0.5)/5.0
                        realB_genA_tr_pesqnorm = ( pesq( real_B_wav , gen_A_wav ) + 0.5)/5.0
                        realB_cycB_tr_pesqnorm = ( pesq( real_B_wav , cyc_B_wav ) + 0.5)/5.0
                        realA_cycA_tr_pesqnorm = ( pesq( real_A_wav , cyc_A_wav ) + 0.5)/5.0
                        ###############################################
                        realB_genB_tr_pesqnorm = ( pesq( real_B_wav , gen_B_wav ) + 0.5)/5.0
                        realA_genA_tr_pesqnorm = ( pesq( real_A_wav , gen_A_wav ) + 0.5)/5.0

                        # print("PESQ Metric is Activated(Scores: -0.5-4.5):")
                        # print("\nrealA_genB_tr_pesqnorm" ,realA_genB_tr_pesqnorm )
                        # print(  "realB_genA_tr_pesqnorm" ,realB_genA_tr_pesqnorm )
                        # print(  "realB_cycB_tr_pesqnorm" ,realB_cycB_tr_pesqnorm )
                        # print(  "realA_cycA_tr_pesqnorm" ,realA_cycA_tr_pesqnorm )                     

                        # print(  "realB_genB_tr_pesqnorm" ,realB_genB_tr_pesqnorm )
                        # print(  "realA_genA_tr_pesqnorm" ,realA_genA_tr_pesqnorm )

                        with open("report_files/realA_genB_tr_pesqnorm.txt", "a") as f05:
                            f05.write( str(realA_genB_tr_pesqnorm ) + "\n" )

                        self.logger._log_scalars( scalar_dict={'A.7_1.Train_realA.genB.PESQnorm':\
                                                               realA_genB_tr_pesqnorm ,
                                                               'A.7_2.Train_realB.genA.PESQnorm':\
                                                               realB_genA_tr_pesqnorm ,
                                                               'A.7_3.Train_realB.cycB.PESQnorm':\
                                                               realB_cycB_tr_pesqnorm ,
                                                               'A.7_4.Train_realA.cycA.PESQnorm':\
                                                               realA_cycA_tr_pesqnorm },
                                                 print_to_stdout = False )


                    if ( Metric_Measure_Active and Metric_Measure_SDR ): 
                        #print("\nMetric_Measure_Active & Metric_Measure_SDR: True")
                        # path = "report_files/wavs/train/"

                        # Clean and den should have the same length, and be 1D
                        #from torchmetrics.audio import SignalDistortionRatio
                        #g = torch.manual_seed(1)
                        #preds = torch.randn(8000)
                        #target = torch.randn(8000)
                        
                        sdr_db = SignalDistortionRatio()
                        #sdr_db = sdr(preds, target)
                        #sdr_mag = 10 ** (sdr_db / 10)

                        realA_genB_tr_sdrnorm = 1.8 * 10 ** ( sdr_db( gen_B_wav , real_A_wav ) / 10.0 ) 
                        realB_genA_tr_sdrnorm = 1.8 * 10 ** ( sdr_db( gen_A_wav , real_B_wav ) / 10.0 ) 
                        realB_cycB_tr_sdrnorm = 1.5 * 10 ** ( sdr_db( cyc_B_wav , real_B_wav ) / 10.0 )
                        realA_cycA_tr_sdrnorm = 1.5 * 10 ** ( sdr_db( cyc_A_wav , real_A_wav ) / 10.0 )
                        ###############################################
                        realB_genB_tr_sdrnorm = 2.0 * 10 ** ( sdr_db( gen_B_wav , real_B_wav ) / 10.0 )
                        realA_genA_tr_sdrnorm = 2.0 * 10 ** ( sdr_db( gen_A_wav , real_A_wav ) / 10.0 )

                        
                        # print("\nrealA_genB_tr_sdrnorm" ,realA_genB_tr_sdrnorm )
                        # print(  "realB_genA_tr_sdrnorm" ,realB_genA_tr_sdrnorm )
                        # print(  "realB_cycB_tr_sdrnorm" ,realB_cycB_tr_sdrnorm )
                        # print(  "realA_cycA_tr_sdrnorm" ,realA_cycA_tr_sdrnorm )                     

                        # print(  "realB_genB_tr_sdrnorm" ,realB_genB_tr_sdrnorm )
                        # print(  "realA_genA_tr_sdrnorm" ,realA_genA_tr_sdrnorm )

                        with open("report_files/realA_genB_tr_sdrnorm.txt", "a") as f06:
                            f06.write( str(realA_genB_tr_sdrnorm ) + "\n" )

                        self.logger._log_scalars( scalar_dict={'A.8_1.Train_realA.genB.SDRnorm':\
                                                               realA_genB_tr_sdrnorm ,
                                                               'A.8_2.Train_realB.genA.SDRnorm':\
                                                               realB_genA_tr_sdrnorm ,
                                                               'A.8_3.Train_realB.cycB.SDRnorm':\
                                                               realB_cycB_tr_sdrnorm ,
                                                               'A.8_4.Train_realA.cycA.SDRnorm':\
                                                               realA_cycA_tr_sdrnorm },
                                                 print_to_stdout = False )

                    #x.requires_grad_(True)                   
                    # optimiz_metric1_genb.requires_grad_(True)
                    # optimiz_metric1_gena.requires_grad_(True)
                    # optimiz_metric1_cycb.requires_grad_(True)
                    # optimiz_metric1_cyca.requires_grad_(True)
                    
                    ####################MB:Preparing for Training with Metric_Optimization of Discriminator #################                
                    if (Metric_Optimize_Active and Metric_Optimize_MMCD and not Metric_Optimize_stoi ):
                        #print("\nMetric_Optimize_Active & True and Metric_Optimize_MMCD:True ")
                        optimization_metric ="MMCD_in_Discr."
                        optimiz_metric1_genb = MMCD_GBTA * mmcdImpact
                        optimiz_metric1_gena = MMCD_GATB * mmcdImpact
                        optimiz_metric1_cycb = MMCD_CBTB * mmcdCycImpact
                        optimiz_metric1_cyca = MMCD_CATA * mmcdCycImpact
                    
                    if (Metric_Optimize_Active and Metric_Optimize_stoi and not Metric_Optimize_MMCD):
                        #print("\nMetric_Optimize_Active & Metric_Optimize_stoi: True")
                        optimization_metric ="STOI_in_Discr."
                        optimiz_metric1_genb = realA_genB_stoi * stoiImpact
                        optimiz_metric1_gena = realB_genA_stoi * stoiImpact
                        optimiz_metric1_cycb = realB_cycB_stoi * stoiCycImpact
                        optimiz_metric1_cyca = realA_cycA_stoi * stoiCycImpact
                        
                    if (Metric_Optimize_Active and Metric_Optimize_MMCD and Metric_Optimize_stoi) :
                        print("\nMulti_Score_Opt.:Metric_Optimize_Active & True and Metric_Optimize_MMCD:True Simultaneously Metric_Optimize_stoi")
                        optimization_metric  ="MMCD_in_Discr. and Metric_Optimize_stoi"
                        optimiz_metric1_genb = 1.0 * ( MMCD_GBTA * 1.5 + realA_genB_stoi * 1.6 ) / 2.0
                        optimiz_metric1_gena = 1.0 * ( MMCD_GATB * 1.5 + realB_genA_stoi * 1.6 ) / 2.0
                        optimiz_metric1_cycb = 1.0 * ( MMCD_CBTB * 1.0 + realB_cycB_stoi * 1.0 ) / 2.0
                        optimiz_metric1_cyca = 1.0 * ( MMCD_CATA * 1.0 + realA_cycA_stoi * 1.0 ) / 2.0
                        
                    if (Metric_Optimize_Active and Metric_Optimize_cosinesim ):
                        #print("Metric_Optimize_Active & True and Metric_Optimiz_cosinesim:True ")
                        optimization_metric  ="COSINESIM_in_Discr."
                        optimiz_metric1_genb = cosinesim_genBtarg * cosineImpact
                        optimiz_metric1_gena = cosinesim_genAtarg * cosineImpact
                        optimiz_metric1_cycb = cosinesim_cycBtarg * cosineCycImpact
                        optimiz_metric1_cyca = cosinesim_cycAtarg * cosineCycImpact
                        #
                        #
                        #
                        # #MB:03.01.20 Loss for SpeakerEmbeddingSimilarityDistance(SESD)
                        # optimiz_metric1_genb = Lsesd_genB* cosineImpact
                        # optimiz_metric1_gena = Lsesd_genA* cosineImpact
                        # optimiz_metric1_cycb = Lsesd_cycB* cosineCycImpact
                        # optimiz_metric1_cyca = Lsesd_cycA* cosineCycImpact
                        #
                        # #MB:03.01.22
                        # optimiz_metric1_genb = SESMetric_genB * cosineImpact
                        # optimiz_metric1_gena = SESMetric_genA * cosineImpact
                        # optimiz_metric1_cycb = SESMetric_cycB * cosineCycImpact
                        # optimiz_metric1_cyca = SESMetric_cycA * cosineCycImpact 
                        #
                        
                        
                    if (Metric_Optimize_Active and Metric_Optimize_nisqa ):
                        #print("Metric_Optimize_Active & True and Metric_Optimize_nisqa:True ")
                        optimization_metric  ="NISQA_in_Discr."
                        optimiz_metric1_genb = Nisqatts_GB_norm * nisqaImpact
                        optimiz_metric1_gena = Nisqatts_GA_norm * nisqaImpact 
                        optimiz_metric1_cycb = Nisqatts_CB_norm * nisqaCycImpact
                        optimiz_metric1_cyca = Nisqatts_CA_norm * nisqaCycImpact
                    
                    if (Metric_Optimize_Active and Metric_Optimize_nisqa and Metric_Optimize_stoi):
                        print("Multi_Score_Opt.:Metric_Optimize_Active & True and Metric_Optimize_nisqa:True ")
                        optimization_metric  ="NISQA_in_Discr."
                        optimiz_metric1_genb = 1.0 * (Nisqatts_GB_norm * nisqaImpact + realA_genB_stoi * 1.4 ) / 2.0
                        optimiz_metric1_gena = 1.0 * (Nisqatts_GA_norm * nisqaImpact + realB_genA_stoi * 1.4 ) / 2.0 
                        optimiz_metric1_cycb = 1.0 * (Nisqatts_CB_norm * nisqaCycImpact + realB_cycB_stoi * 1.0 ) / 2.0
                        optimiz_metric1_cyca = 1.0 * (Nisqatts_CA_norm * nisqaCycImpact + realA_cycA_stoi * 1.0 ) / 2.0
                        
                                        
                    if (Metric_Optimize_Active and Metric_Optimize_IQAmetric ):
                        #print(f'\nIQA Metric Optimizing is Activated')
                        optimiz_metric1_genb = 1.0 * IQAmetric1genB * IQAImpact 
                        optimiz_metric1_gena = 1.0 * IQAmetric1genA * IQAImpact  
                        optimiz_metric1_cycb = 1.0 * IQAmetric1cycB * IQACycImpact
                        optimiz_metric1_cyca = 1.0 * IQAmetric1cycA * IQACycImpact
                            
                        
                    if (Metric_Optimize_Active and Metric_Optimize_pesq ):
                        #print("\nMetric_Optimize_Active & Metric_Optimize_pesq: True")
                        optimization_metric  ="PESQ_in_Discr."
                        optimiz_metric1_genb = realA_genB_tr_pesqnorm * pesqImpact
                        optimiz_metric1_gena = realB_genA_tr_pesqnorm * pesqImpact
                        optimiz_metric1_cycb = realB_cycB_tr_pesqnorm * 1.0
                        optimiz_metric1_cyca = realA_cycA_tr_pesqnorm * 1.0 
                    
                    if (Metric_Optimize_Active and Metric_Optimize_SDR ):        
                        #print("\nMetric_Optimize_Active & Metric_Optimize_SDR: True")
                        optimization_metric  ="SDR_in_Discr."
                        optimiz_metric1_genb = realA_genB_tr_sdrnorm * sdrImpact
                        optimiz_metric1_gena = realB_genA_tr_sdrnorm * sdrImpact
                        optimiz_metric1_cycb = realB_cycB_tr_sdrnorm * 1.0
                        optimiz_metric1_cyca = realA_cycA_tr_sdrnorm * 1.0 
                    
                    # with torch.set_grad_enabled(True):
                    #     optimiz_metric1_genb = optimiz_metric1_genb.to(self.device, dtype=torch.float)
                    #     optimiz_metric1_gena= optimiz_metric1_2.to(self.device, dtype=torch.float)
                    #     optimiz_metric1_cycb = optimiz_metric1_3.to(self.device, dtype=torch.float)
                    #     optimiz_metric1_cyca = optimiz_metric1_4.to(self.device, dtype=torch.float)                   
                    #     #mask_A = mask_A.to(self.device, dtype=torch.float)  
                    
                    
                    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
                    #if Metric_Optimize_Active_inDisc :
                    if Metric_Optimize_Active :
                        metric_sco_pred_genA = torch.mean( d_fake_A   )
                        metric_sco_pred_genB = torch.mean( d_fake_B   )
                        metric_sco_pred_cycA = torch.mean( d_cycled_A )
                        metric_sco_pred_cycB = torch.mean( d_cycled_B )
                        #----------------------------------------------
                        metric_sco_pred_realA = torch.mean( d_real_A )
                        metric_sco_pred_realB = torch.mean( d_real_B )
                        #------------------------------------------------
                        genAmetricLoss = torch.abs(optimiz_metric1_gena - metric_sco_pred_genA)**2
                        genBmetricLoss = torch.abs(optimiz_metric1_genb - metric_sco_pred_genB)**2
                        cycAmetricLoss = torch.abs(optimiz_metric1_cyca - metric_sco_pred_cycA)**2
                        cycBmetricLoss = torch.abs(optimiz_metric1_cycb - metric_sco_pred_cycB)**2
                        #pass
                    
                        gen_distanceA = 1 * torch.abs(optimiz_metric1_gena  - metric_sco_pred_genA)
                        gen_distanceB = 1 * torch.abs(optimiz_metric1_genb  - metric_sco_pred_genB)
                        cyc_distanceA = 1 * torch.abs(optimiz_metric1_cyca  - metric_sco_pred_cycA)
                        cyc_distanceB = 1 * torch.abs(optimiz_metric1_cycb  - metric_sco_pred_cycB)
                        #
                        # distA = (0 * gen_distanceA + 0.0 * cyc_distanceA)/2.0
                        # distB = (0 * gen_distanceB + 0.0 * cyc_distanceB)/2.0
                        # metric_dist = 5.0 * ( distA + distB ) / 2.0
                        #
                        self.logger._log_scalars( scalar_dict={ 'A.9_1.Train_optimiz_metric1_genb':optimiz_metric1_genb,
                                                                'A.9_2.Train_optimiz_metric1_gena':optimiz_metric1_gena,
                                                                'A.9_3.Train_optimiz_metric1_cyca':optimiz_metric1_cyca,
                                                                'A.9_4.Train_optimiz_metric1_cycb':optimiz_metric1_cycb,
                                                               },
                                                 print_to_stdout = False )
                        self.logger._log_scalars( scalar_dict={ 'A.10_1.Train_metric_sco_pred_genB':metric_sco_pred_genB,
                                                                'A.10_2.Train_metric_sco_pred_genA':metric_sco_pred_genA,
                                                                'A.10_3.Train_metric_sco_pred_cycB':metric_sco_pred_cycB,
                                                                'A.10_4.Train_metric_sco_pred_cycA':metric_sco_pred_cycA,
                                                               },
                                                 print_to_stdout = False )
                        self.logger._log_scalars( scalar_dict={ 'A.11_1.Train_gen_distanceB':gen_distanceB,
                                                                'A.11_2.Train_gen_distanceA':gen_distanceA,
                                                                'A.11_3.Train_cyc_distanceB':cyc_distanceB,
                                                                'A.11_4.Train_cyc_distanceA':cyc_distanceA,
                                                               },
                                                 print_to_stdout = False )


                    # Loss Functions
                    d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                    # d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)    
                    #optmute = 1.0
                    #print("***** Metric_Optimize_Active***** is:",Metric_Optimize_Active)
                    if (Metric_Optimize_Active and 0):
                        #d_loss_A_fake = torch.mean((optmute*optimiz_metric1_gena - damp_DFA*d_fake_A)**2)+ extraLoss_fa*(1-optimiz_metric1_gena)**2
                        d_loss_A_fake = torch.mean((0*optmute*optimiz_metric1_gena - damp_DFA*d_fake_A)**2)+ extraLoss_fa*(1-optimiz_metric1_gena)**2 +0*genAmetricLoss**2
                        
                    else:
                        d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                        #print("\n**Conventional Mask Loss")
                    
                    #MB:02.08.27:only for test
                    # d_loss_A_fake0 = torch.mean((0 - d_fake_A) ** 2)   
                    # print("\noptimiz_metric1_gena:",optimiz_metric1_gena)
                    # print("extraLoss_fa: ",extraLoss_fa)
                    # print("extraLoss_fa*(1-optimiz_metric1_gena)**2",extraLoss_fa*(1-optimiz_metric1_gena)**2)
                    # print("\nd_loss_A_fake0:",d_loss_A_fake0)
                    # print("d_loss_A_fake :",d_loss_A_fake)
                    
                    #d_loss_A_fake = torch.mean((realB_genA_stoi - d_fake_A) ** 2)      
                    d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0
                    d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                    
                    # d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                    
                    if (Metric_Optimize_Active and 0):
                        #d_loss_B_fake = torch.mean((optmute*optimiz_metric1_genb - damp_DFB*d_fake_B)**2) + extraLoss_fb*(1-optimiz_metric1_genb)**2
                        d_loss_B_fake = torch.mean((0*optmute*optimiz_metric1_genb - damp_DFB*d_fake_B)**2) + extraLoss_fb*(1-optimiz_metric1_genb)**2 +0*genBmetricLoss**2
                        #print("****optimiz_metric1_genb(MMCD.Opt: MMCD_GBTA)=",optimiz_metric1_genb)
                        
                    else:
                        d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                        #print("\n********Conventional Mask Loss")
                    
                    #MB:02.08.27:only for test
                    # d_loss_B_fake0 = torch.mean((0 - d_fake_B) ** 2)   
                    # print("d_loss_B_fake0:",d_loss_B_fake0)
                    # print("d_loss_B_fake :",d_loss_B_fake)
                        
                    d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                    # Two Step Adverserial Loss
                    # d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2) 
                    if (Metric_Optimize_Active and 0):
                        #d_loss_A_cycled = torch.mean((optmute*optimiz_metric1_cyca - damp_DCA*d_cycled_A)**2) + extraLoss_ca*(1-optimiz_metric1_cyca)**2
                        d_loss_A_cycled = torch.mean((0 * optmute*optimiz_metric1_cyca - damp_DCA*d_cycled_A)**2) + extraLoss_ca*(1-optimiz_metric1_cyca)**2 + 5*cycAmetricLoss**2      
                    else:
                        d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)  
                    #MB:02.08.27:only for test
                    # d_loss_A_cycled0 = torch.mean((0 - d_cycled_A) ** 2)   
                    # print("d_loss_A_cycled0:",d_loss_A_cycled0)
                    # print("d_loss_A_cycled :",d_loss_A_cycled)
                        
                    # d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                    if (Metric_Optimize_Active and 0):
                        #d_loss_B_cycled = torch.mean((optmute*optimiz_metric1_cycb - damp_DCB*d_cycled_B)**2) + extraLoss_cb*(1-optimiz_metric1_cycb)**2
                        d_loss_B_cycled = torch.mean((0 * optmute*optimiz_metric1_cycb - damp_DCB*d_cycled_B)**2) + extraLoss_cb*(1-optimiz_metric1_cycb)**2 + 5*cycBmetricLoss**2        
                    else:
                        d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)  
                    #MB:02.08.27:only for test
                    # d_loss_B_cycled0 = torch.mean((0 - d_cycled_B) ** 2)   
                    # print("d_loss_B_cycled0:",d_loss_B_cycled0)
                    # print("d_loss_B_cycled :",d_loss_B_cycled)
                        
                    
                           
                    #d_loss_B_cycled = torch.mean((realB_cycB_stoi - d_cycled_B) ** 2)  #MB:02.05.05
                    d_loss_A2_real = torch.mean( (1 - torch.mean(d_real_A2) )** 2)
                    d_loss_B2_real = torch.mean( (1 - torch.mean(d_real_B2) )** 2)
                    d_loss_A_2nd = (1.0*d_loss_A2_real + 1.0*d_loss_A_cycled) / 2.0 
                    d_loss_B_2nd = (1.0*d_loss_B2_real + 1.0*d_loss_B_cycled) / 2.0
                    
                    # genAmetricLoss = torch.abs(optimiz_metric1_gena - metric_sco_pred_genA)**2
                    # genBmetricLoss = torch.abs(optimiz_metric1_genb - metric_sco_pred_genB)**2
                    # cycAmetricLoss = torch.abs(optimiz_metric1_cyca - metric_sco_pred_cycA)**2
                    # cycBmetricLoss = torch.abs(optimiz_metric1_cycb - metric_sco_pred_cycB)**2

                    
                    
                    
                    # if self.logger.epoch    < 246: 
                    #     gamma = 0
                    #     tetta = 8.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 246:
                    #     gamma = 0
                    #     tetta = 8.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 400:
                    #     gamma = 0
                    #     tetta = 10.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 600:
                    #     gamma = 0
                    #     tetta = 15.0
                    #     zetta = 0.0
                    # else:
                    #     pass
                    
                    # if self.logger.epoch      <= 200: 
                    #     gamma = 1.0
                    #     tetta = 0.0
                    #     etta  = 1.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  <= 400:
                    #     gamma = 1.0
                    #     tetta = 0.0
                    #     ettaa = 1.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  <= 600:
                    #     gamma = 1.0
                    #     tetta = 0.0
                    #     etta  = 1.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  <= 800:
                    #     gamma = 1.0
                    #     tetta = 0.0
                    #     etta  = 1.0
                    #     zetta = 0.0
                    # else:
                    #     pass
                    
                    # if self.logger.epoch    < 246: 
                    #     etta  = 1.0 
                    #     gamma = 0.0
                    #     tetta = 10
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 246:
                    #     etta  = 1.0
                    #     gamma = 1.0
                    #     tetta = 8
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 400:
                    #     etta  = 1.0
                    #     gamma = 1.0
                    #     tetta = 10.0
                    #     zetta = 0.0
                    # elif self.logger.epoch  >= 600:
                    #     etta  = 1.0
                    #     gamma = 5.0
                    #     tetta = 20.0
                    #     zetta = 0.0
                    # else:
                    #     pass
                    
                    ############## MB: 03.02.16 ##################
                    gamma = 1.0
                    tetta = 10.0
                    etta =  1.0
                    zetta = 4.0   
                    
                    if (not Metric_Optimize_Active ):
                        genAmetricLoss = 0.0
                        genBmetricLoss = 0.0 
                        cycAmetricLoss = 0.0 
                        cycBmetricLoss = 0.0 
                      
                    # Final Loss for discriminator with the Two Step Adverserial Loss
                    d_loss = etta*(d_loss_A + d_loss_B) / 2.0  + zetta*(genAmetricLoss + genBmetricLoss ) /2.0 + \
                    gamma*(d_loss_A_2nd + d_loss_B_2nd) / 2.0  + tetta*(cycAmetricLoss + cycBmetricLoss ) /2.0

                    # Backprop for Discriminator
                    self.reset_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                                        
                    # ----------------
                    # Train Generator
                    # ----------------
                    self.generator_A2B.train()
                    self.generator_B2A.train()
                    self.discriminator_A.eval()
                    self.discriminator_B.eval()
                    self.discriminator_A2.eval()
                    self.discriminator_B2.eval()

                    # Generator Feed Forward
                    fake_B = self.generator_A2B(real_A, mask_A)
                    cycle_A = self.generator_B2A(fake_B, torch.ones_like(fake_B))
                    fake_A = self.generator_B2A(real_B, mask_B)
                    cycle_B = self.generator_A2B(fake_A, torch.ones_like(fake_A))
                    identity_A = self.generator_B2A(
                        real_A, torch.ones_like(real_A))
                    identity_B = self.generator_A2B(
                        real_B, torch.ones_like(real_B))
                    d_fake_A = self.discriminator_A(fake_A)
                    d_fake_B = self.discriminator_B(fake_B)

                    ################ Fabricated Synthesized Files in Generators  ######################
                    ####################
                    # Measurements in Generator
                    if(Metric_Measure_Active and Metric_Measure_MMCD and False ): 
                        pass
                        
                    # Measurements in Generator
                    if ( Metric_Measure_Active and Metric_Measure_stoi and False ):            
                        pass
                        
                    # Measurements in Generator
                    if ( Metric_Measure_Active and Metric_Measure_cosinesim and False ): 
                        pass
                    # Measurements in Generator
                    if ( Metric_Measure_Active and Metric_Measure_nisqa and False ): 
                        pass                                                           
                    ##################################################################################
                    
                    # For Two Step Adverserial Loss
                    d_fake_cycle_A = self.discriminator_A2(cycle_A)
                    d_fake_cycle_B = self.discriminator_B2(cycle_B)

                    # Generator Cycle Loss
                    cycleLoss = torch.mean(
                        torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))
                    
                    # Generator Identity Loss
                    identityLoss = torch.mean(
                        torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                    # Generator Loss
                    g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                    g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                    # Generator Two Step Adverserial Loss
                    generator_loss_A2B_2nd = torch.mean((1 - d_fake_cycle_B) ** 2)
                    generator_loss_B2A_2nd = torch.mean((1 - d_fake_cycle_A) ** 2)
                    
 
                    # if self.logger.epoch   <= 200: 
                    #     alpha = 0.4
                    #     betta = 0.8
                    # elif self.logger.epoch <= 400:
                    #     alpha = 0.4
                    #     betta = 0.8
                    # elif self.logger.epoch <= 600:
                    #     alpha = 0.2
                    #     betta = 0.8
                    # elif self.logger.epoch  > 600:
                    #     alpha = 0.2
                    #     betta = 0.8
                    # else:
                    #     pass
                    
                    # if self.logger.epoch   <= 200: 
                    #     alpha = 1.0
                    #     betta = 0.8
                    #     kappa = 1.0
                    # elif self.logger.epoch <= 400:
                    #     alpha = 0.7
                    #     betta = 0.8
                    #     kappa = 1.0
                    # elif self.logger.epoch <= 600:
                    #     alpha = 0.4
                    #     betta = 0.4
                    #     kappa = 1.0
                    # elif self.logger.epoch  > 600:
                    #     alpha = 0.4
                    #     betta = 0.8
                    #     kappa = 1.0
                    # else:
                    #     pass
                    
                    #alpha = 1.0
                    alpha = 0.2
                    #betta = 1.0
                    betta =  0.8
                    kappa = 1.0
                
                    # Total Generator Loss
                    #MB:03.01.11
                    g_loss = (g_loss_A2B + g_loss_B2A) + \
                        kappa*(generator_loss_A2B_2nd + generator_loss_B2A_2nd) + \
                        alpha * self.cycle_loss_lambda * cycleLoss + betta * self.identity_loss_lambda * identityLoss
                        #self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss

                    # Backprop for Generator
                    self.reset_grad()
                    g_loss.backward()
                    self.generator_optimizer.step()
                    
                                        
                    #########03.02.16 2nd train of Disc%%%%%%%%%%%
                    # ----------------------
                    # Train Discriminator @Again
                    # ----------------------
                    self.generator_A2B.eval()
                    self.generator_B2A.eval()
                    self.discriminator_A.train()
                    self.discriminator_B.train()
                    self.discriminator_A2.train()
                    self.discriminator_B2.train()

                    # Discriminator Feed Forward
                    d_real_A = self.discriminator_A(real_A)
                    d_real_B = self.discriminator_B(real_B)
                    d_real_A2 = self.discriminator_A2(real_A)
                    d_real_B2 = self.discriminator_B2(real_B)
                    generated_A = self.generator_B2A(real_B, mask_B)
                    d_fake_A = self.discriminator_A(generated_A)

                    # For Two Step Adverserial Loss A->B
                    cycled_B = self.generator_A2B(
                        generated_A, torch.ones_like(generated_A))
                    d_cycled_B = self.discriminator_B2(cycled_B)

                    generated_B = self.generator_A2B(real_A, mask_A)
                    d_fake_B = self.discriminator_B(generated_B)

                    # For Two Step Adverserial Loss B->A
                    cycled_A = self.generator_B2A(
                        generated_B, torch.ones_like(generated_B))
                    d_cycled_A = self.discriminator_A2(cycled_A)
                   
                    if 1:
                        d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                        d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                        #
                        d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0
                        #
                        d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                        d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                        #
                        d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0
                        #
                        d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)  
                        d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)  
                    
                           
                    #d_loss_B_cycled = torch.mean((realB_cycB_stoi - d_cycled_B) ** 2)  #MB:02.05.05
                    d_loss_A2_real = torch.mean((1 - d_real_A2) ** 2)
                    d_loss_B2_real = torch.mean((1 - d_real_B2) ** 2)
                    #
                    ############## MB: 03.02.16 ##################
                    d_loss_A_2nd = (1.0*d_loss_A2_real + 1.0*d_loss_A_cycled) / 2.0 
                    d_loss_B_2nd = (1.0*d_loss_B2_real + 1.0*d_loss_B_cycled) / 2.0
                    
                    ############## MB: 03.02.16 ##################
                    etta = 1.0
                    #gamma = 1.0 
                    gamma = 0.4
                    # Final Loss for discriminator with the Two Step Adverserial Loss
                    d_loss = etta*(d_loss_A + d_loss_B) / 2.0 +\
                    gamma*(d_loss_A_2nd + d_loss_B_2nd) / 2.0  

                    # Backprop for Discriminator
                    self.reset_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                    
                # Log Iteration on Tensorboard
                self.logger.log_iter(
                    loss_dict={'A.1_g_lossTrain': g_loss.item(), 'A.2_d_lossTrain': d_loss.item()})
                self.logger.end_iter()

                # Adjust learning rates
                if self.logger.global_step > self.decay_after:
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=True)
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=False)

                # Set identity loss to zero if larger than given value
                if self.logger.global_step > self.stop_identity_after:
                    self.identity_loss_lambda = 0

            if (Metric_ShowPerEpoch_Active and Metric_Optimize_Active and (epoch_number % 10 == 0) ):
                    #print("\noptimization_metric:",optimization_metric,"The last iteration of Epoch_Number:",epoch_number ) 
                    print("optimiz_metric1_genb", optimiz_metric1_genb )
                    print("optimiz_metric1_gena", optimiz_metric1_gena )
                    print("optimiz_metric1_cycb", optimiz_metric1_cycb )
                    print("optimiz_metric1_cyca", optimiz_metric1_cyca )
            
            # Log intermediate outputs on Tensorboard
            if self.logger.epoch % self.epochs_per_plot == 0 and 0 :
                with torch.no_grad():
                    # Log Mel-spectrograms .png
                    #pass
                    real_mel_A_fig = get_mel_spectrogram_fig(
                        real_A[0].detach().cpu())
                    fake_mel_A_fig = get_mel_spectrogram_fig(
                        generated_A[0].detach().cpu())
                    real_mel_B_fig = get_mel_spectrogram_fig(
                        real_B[0].detach().cpu())
                    fake_mel_B_fig = get_mel_spectrogram_fig(
                        generated_B[0].detach().cpu())
                    
                    self.logger.visualize_outputs({"real_A_spec": real_mel_A_fig, "fake_B_spec": fake_mel_B_fig,
                                                   "real_B_spec": real_mel_B_fig, "fake_A_spec": fake_mel_A_fig})

                    # Convert Mel-spectrograms from validation set to waveform and log to tensorboard
                    real_mel_full_A, real_mel_full_B = next(
                        iter(self.validation_dataloader))
                    real_mel_full_A = real_mel_full_A.to(
                        self.device, dtype=torch.float)
                    real_mel_full_B = real_mel_full_B.to(
                        self.device, dtype=torch.float)
                    fake_mel_full_B = self.generator_A2B(
                        real_mel_full_A, torch.ones_like(real_mel_full_A))
                    fake_mel_full_A = self.generator_B2A(
                        real_mel_full_B, torch.ones_like(real_mel_full_B))
                    real_wav_full_A = decode_melspectrogram(self.vocoder, real_mel_full_A[0].detach(
                    ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                    fake_wav_full_A = decode_melspectrogram(self.vocoder, fake_mel_full_A[0].detach(
                    ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                    
                    real_wav_full_B = decode_melspectrogram(self.vocoder, real_mel_full_B[0].detach(
                    ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                    fake_wav_full_B = decode_melspectrogram(self.vocoder, fake_mel_full_B[0].detach(
                    ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                    self.logger.log_audio(
                        real_wav_full_A.T, "real_speaker_A_audio", self.sample_rate)
                    self.logger.log_audio(
                        fake_wav_full_A.T, "fake_speaker_A_audio", self.sample_rate)
                    self.logger.log_audio(
                        real_wav_full_B.T, "real_speaker_B_audio", self.sample_rate)
                    self.logger.log_audio(
                        fake_wav_full_B.T, "fake_speaker_B_audio", self.sample_rate)
                    

            # Save each model checkpoint
            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.generator_A2B,
                                self.generator_optimizer, None, args.device, "generator_A2B")
                self.saver.save(self.logger.epoch, self.generator_B2A,
                                self.generator_optimizer, None, args.device, "generator_B2A")
                self.saver.save(self.logger.epoch, self.discriminator_A,
                                self.discriminator_optimizer, None, args.device, "discriminator_A")
                self.saver.save(self.logger.epoch, self.discriminator_B,
                                self.discriminator_optimizer, None, args.device, "discriminator_B")
                self.saver.save(self.logger.epoch, self.discriminator_A2,
                                self.discriminator_optimizer, None, args.device, "discriminator_A2")
                self.saver.save(self.logger.epoch, self.discriminator_B2,
                                self.discriminator_optimizer, None, args.device, "discriminator_B2")
            
            #print("\nrealA_genB_stoi:",realA_genB_stoi,"\n81th file of epoch(self.logger.epoch)",self.logger.epoch)  #MB:02.05.05
            # self.logger.end_epoch()
            
            # tracking validation for (real_A, real_B) in enumerate(tqdm(self.validation_dataloader)):
            #if self.logger.epoch % self.epochs_per_save == 0:
           
            #mmcd_test_mean      = 0.0
            mcd_test_mean      = 0.0
            mcd_test_realAfakeBmean = 0.0
            
            mcd_test_mean       = 0.0
            stoi_test_mean      = 0.0
            stoi_test_realAfakeBmean = 0.0
            
            cosinesim_test_mean = 0.0
            nisqa_test_mean     = 0.0
            pesq_test_mean      = 0.0
            sdr_test_mean       = 0.0
                    
            if self.logger.epoch %  tbTestLogPerEpo == 0 or self.logger.epoch == 1 or self.logger.epoch == (PercepOptActivationEpoch1+1) or self.logger.epoch==( PercepOptActivationEpoch2 +1 ) : 
                d_loss_list = []
                g_loss_list = []
                
                stoi_test_list      = []
                realA_fakeB_stoi_list=[]
                #mmcd_test_list      = []
                mcd_test_list      = []
                
                realAfakeB_mcd_list = []
                
                cosinesim_test_list = []
                nisqa_test_list     = []
                pesq_test_list      = []
                sdr_test_list       = []
                
                #for i, (real_mel_full_A, real_mel_full_B) in enumerate(tqdm(self.validation_dataloader)):
                for i, (real_mel_full_A, real_mel_full_B) in enumerate(tqdm(self.test_dataloader)):
                    
                    #print("\n",real_mel_full_A.size())
                    #print("\n",real_mel_full_B.size())
                    #print("\nith file of evaluation(TEST Files): =",i+1)

                    with torch.no_grad():
                        #For DTW 02.07.11
                        # real_me_full_A0 = real_mel_full_A
                        # real_mel_full_B0 = real_mel_full_B
                        
                        real_mel_full_A = real_mel_full_A.to(
                            self.device, dtype=torch.float)
                        real_mel_full_B = real_mel_full_B.to(
                            self.device, dtype=torch.float)
                        
                        # Generator Feed Forward
                        fake_mel_full_B = self.generator_A2B(real_mel_full_A, torch.ones_like(real_mel_full_A))
                        cycle_mel_full_A = self.generator_B2A(fake_mel_full_B, torch.ones_like(fake_mel_full_B))

                        #print("\nreal_mel_full_B.size()",real_mel_full_B.size())
                        fake_mel_full_A = self.generator_B2A(real_mel_full_B, torch.ones_like(real_mel_full_B))
                        #print("\nfake_mel_full_A.size()",fake_mel_full_A.size())
                        cycle_mel_full_B = self.generator_A2B(fake_mel_full_A, torch.ones_like(fake_mel_full_A))
                        #print("\ncycle_mel_full_B.size()",cycle_mel_full_B.size())


                        #print("\n\nAfter ",real_mel_full_B.size() )
                        #print("\n\nAfter",cycle_mel_full_B.size())

                        identity_full_A = self.generator_B2A(real_mel_full_A, torch.ones_like(real_mel_full_A))
                        identity_full_B = self.generator_A2B(real_mel_full_B, torch.ones_like(real_mel_full_B))

                        d_fake_full_A = self.discriminator_A(fake_mel_full_A)
                        d_fake_full_B = self.discriminator_B(fake_mel_full_B)


                        # For DTW 02.07.11
                        # fake_mel_full_B0  = fake_mel_full_B
                        # fake_mel_full_A0  = fake_mel_full_A 
                        
                        
                        # For Two Step Adverserial Loss
                        d_fake_cycle_full_A = self.discriminator_A2(cycle_mel_full_A)
                        d_fake_cycle_full_B = self.discriminator_B2(cycle_mel_full_B)

                        minimum = min(real_mel_full_B.size()[2], cycle_mel_full_B.size()[2])
                        real_mel_full_B =real_mel_full_B[:,:,:minimum]
                        cycle_mel_full_B = cycle_mel_full_B[:,:,:minimum]

                        minimum = min(real_mel_full_A.size()[2], cycle_mel_full_A.size()[2])
                        real_mel_full_A =real_mel_full_A[:,:,:minimum]
                        cycle_mel_full_A = cycle_mel_full_A[:,:,:minimum]

                        # Generator Cycle Loss
                        cycleLossFull = torch.mean(torch.abs(real_mel_full_A - cycle_mel_full_A)) + \
                                    torch.mean(torch.abs(real_mel_full_B - cycle_mel_full_B))


                        # Generator Identity Loss
                        minimum = min(real_mel_full_A.size()[2], identity_full_A.size()[2])
                        real_mel_full_A =real_mel_full_A[:,:,:minimum]
                        identity_full_A = identity_full_A[:,:,:minimum]

                        minimum = min(real_mel_full_B.size()[2], identity_full_B.size()[2])
                        real_mel_full_B = real_mel_full_B[:,:,:minimum]
                        identity_full_B = identity_full_B[:,:,:minimum]


                        identityLossFull = torch.mean(torch.abs(real_mel_full_A - identity_full_A)) + \
                                        torch.mean(torch.abs(real_mel_full_B - identity_full_B))


                        g_loss_A2B_full = torch.mean((1 - d_fake_full_B) ** 2)
                        g_loss_B2A_full = torch.mean((1 - d_fake_full_A) ** 2)


                        # Generator Two Step Adverserial Loss
                        generator_loss_A2B_2nd_full = torch.mean((1 - d_fake_cycle_full_B) ** 2)
                        generator_loss_B2A_2nd_full = torch.mean((1 - d_fake_cycle_full_A) ** 2)


                        # Total Generator Loss
                        g_loss_full = g_loss_A2B_full + g_loss_B2A_full + \
                            generator_loss_A2B_2nd_full + generator_loss_B2A_2nd_full + \
                            self.cycle_loss_lambda * cycleLossFull + self.identity_loss_lambda * identityLossFull


                        # Discriminator Feed Forward
                        d_real_full_A = self.discriminator_A(real_mel_full_A)
                        d_real_full_B = self.discriminator_B(real_mel_full_B)
                        d_real_full_A2 = self.discriminator_A2(real_mel_full_A)
                        d_real_full_B2 = self.discriminator_B2(real_mel_full_B)
                        generated_mel_full_A = self.generator_B2A(real_mel_full_B, torch.ones_like(real_mel_full_B))
                        d_fake_full_A = self.discriminator_A(generated_mel_full_A)   #TODO Delete: Repeated


                        # For Two Step Adverserial Loss A->B
                        cycled_mel_full_B = self.generator_A2B(generated_mel_full_A, torch.ones_like(generated_mel_full_A))
                        d_cycled_full_B = self.discriminator_B2(cycled_mel_full_B)

                        generated_mel_full_B = self.generator_A2B(real_mel_full_A, torch.ones_like(real_mel_full_A))
                        d_fake_full_B = self.discriminator_B(generated_mel_full_B)   #TODO Delete: Repeated

                        # For Two Step Adverserial Loss B->A
                        cycled_mel_full_A = self.generator_B2A(generated_mel_full_B, torch.ones_like(generated_mel_full_B))  #TODO Delete: Repeated
                        d_cycled_full_A = self.discriminator_A2(cycled_mel_full_A) #TODO Delete: Repeated

                        #for DTW 02.07.11
                        # cycled_mel_full_A0 = cycled_mel_full_A
                        # cycled_mel_full_B0 = cycled_mel_full_B                       
                        
                        # Loss Functions
                        d_loss_A_real_full = torch.mean((1 - d_real_full_A) ** 2)
                        d_loss_A_fake_full = torch.mean((0 - d_fake_full_A) ** 2)
                        
                        d_loss_A_full      = (d_loss_A_real_full + d_loss_A_fake_full) / 2.0

                        d_loss_B_real_full = torch.mean((1 - d_real_full_B) ** 2)
                        d_loss_B_fake_full = torch.mean((0 - d_fake_full_B) ** 2)
                        
                        d_loss_B_full      = (d_loss_B_real_full + d_loss_B_fake_full) / 2.0

                        # Two Step Adverserial Loss
                        d_loss_A_cycled_full = torch.mean((0 - d_cycled_full_A) ** 2)
                        d_loss_B_cycled_full = torch.mean((0 - d_cycled_full_B) ** 2)
                        d_loss_A2_real_full  = torch.mean((1 - d_real_full_A2) ** 2)
                        d_loss_B2_real_full  = torch.mean((1 - d_real_full_B2) ** 2)
                        d_loss_A_2nd_full    = (d_loss_A2_real_full + d_loss_A_cycled_full) / 2.0
                        d_loss_B_2nd_full    = (d_loss_B2_real_full + d_loss_B_cycled_full) / 2.0

                        # Final Loss for discriminator with the Two Step Adverserial Loss
                        d_loss_full = (d_loss_A_full + d_loss_B_full) / 2.0 + \
                            (d_loss_A_2nd_full + d_loss_B_2nd_full) / 2.0

                
                        
                        # Generating Audio Files for Test Dataset  
                        #if ( Metric_Measure_Test_Active ): 
                        if self.logger.epoch % 1 == 0: 
                            
                            real_wav_full_A = decode_melspectrogram(self.vocoder, real_mel_full_A[0].detach(
                            ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                            #print(f'\n\n{real_wav_full_A}')
                            fake_wav_full_A = decode_melspectrogram(self.vocoder, fake_mel_full_A[0].detach(
                            ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                            #).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu() corrected 02.08.25
                            real_wav_full_B = decode_melspectrogram(self.vocoder, real_mel_full_B[0].detach(
                            ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                            fake_wav_full_B = decode_melspectrogram(self.vocoder, fake_mel_full_B[0].detach(
                            ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                            #).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu() rectified 02.08.25
                            
                            #MB:02.09.02: self.dataset_A_mean  -> self.test_dataset_A_std
                            #MB:02.09.02: self.dataset_B_mean  -> self.test_dataset_B_std
                            #MB:02.09.02: self.dataset_B_std   -> self.test_dataset_A_std
                            #MB:02.09.02: self.dataset_B_std   -> self.test_dataset_B_std
                            # real_wav_full_A = decode_melspectrogram(self.vocoder, real_mel_full_A[0].detach(
                            # ).cpu(), self.test_dataset_A_mean, self.test_dataset_A_std).cpu()
                            # #print(f'\n\n{real_wav_full_A}')
                            # fake_wav_full_A = decode_melspectrogram(self.vocoder, fake_mel_full_A[0].detach(
                            # ).cpu(), self.test_dataset_A_mean, self.test_dataset_A_std).cpu()
                            # #).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu() corrected 02.08.25
                            # real_wav_full_B = decode_melspectrogram(self.vocoder, real_mel_full_B[0].detach(
                            # ).cpu(), self.test_dataset_B_mean, self.test_dataset_B_std).cpu()
                            # fake_wav_full_B = decode_melspectrogram(self.vocoder, fake_mel_full_B[0].detach(
                            # ).cpu(), self.test_dataset_B_mean, self.test_dataset_B_std).cpu()
                            
                            
                            #For DTW 02.07.11
                            # real_mel_full_A0 
                            # real_mel_full_B0 
                            # fake_mel_full_B0
                            # fake_mel_full_A0
                            # cycled_mel_full_A0
                            # cycled_mel_full_B0

                            #Time_Align_TestFiles = False
                            if Time_Align_TestFiles == True :
                                # pad 0
                                # if len(loaded_ref_wav)<len(loaded_syn_wav):
                                #     loaded_ref_wav = np.pad(loaded_ref_wav, (0, len(loaded_syn_wav)-len(loaded_ref_wav)))
                                #     else:
                                #     loaded_syn_wav = np.pad(loaded_syn_wav, (0, len(loaded_ref_wav)-len(loaded_syn_wav)))

                                # extract MCEP features (vectors): 2D matrix (num x mcep_size)
                                # ref_mcep_vec = self.wav2mcep_numpy(loaded_ref_wav)
                                # syn_mcep_vec = self.wav2mcep_numpy(loaded_syn_wav)

                                #_, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)

                                # _, pathB = fastdtw(real_mel_full_B0[:, :], fake_mel_full_B0[:, :], dist=euclidean)
                                # _, pathA = fastdtw(real_mel_full_A0[:, :], fake_mel_full_A0[:, :], dist=euclidean)

                                import librosa
                                import math
                                import numpy as np
                                import pyworld
                                import pysptk
                                from fastdtw import fastdtw
                                from scipy.spatial.distance import euclidean
                                
                                # print("\nreal_mel_full_B[0].T.shape",real_mel_full_B[0].T.shape)
                                # print("\nreal_mel_full_A[0].T.shape",real_mel_full_A[0].T.shape)
                                # print("\nfake_mel_full_B[0].T.shape",fake_mel_full_B[0].T.shape)
                                # print("\nfake_mel_full_A[0].T.shape",fake_mel_full_A[0].T.shape)
                            
                                x = real_mel_full_B.squeeze().T
                                y = fake_mel_full_B.squeeze().T
                                _, pathB = fastdtw(x[:, :].cpu(), y[:, :].cpu(), dist=euclidean)
                                
                                p = real_mel_full_A.squeeze().T
                                q = fake_mel_full_A.squeeze().T
                                _, pathA = fastdtw(p[:, :].cpu(), q[:, :].cpu(), dist=euclidean)                 
                                #print("\npathB",pathB)

                                '''
                                param path: pairs between x and y
                                '''
                                # x = real_mel_full_B0
                                # y = fake_mel_full_B0
                                # p = real_mel_full_A0
                                # q = fake_mel_full_A0

                                pathBx = list(map(lambda l: l[0], pathB))
                                pathBy = list(map(lambda l: l[1], pathB))
                                x1, y1 = x[pathBx], y[pathBy]
                                
                                
                                pathAp = list(map(lambda l: l[0], pathA))
                                pathAq = list(map(lambda l: l[1], pathA))
                                p1, q1 = p[pathAp], q[pathAq]
                                
                                # print("\nx1.shape",x1.shape)
                                # print("\ny1.shape",y1.shape)
                                
                                # print("\np",p.shape)
                                # print("\nq",q.shape)

                                #frames_tot = x.shape[0]       # length of pairs
                                # print("\nself.dataset_A_mean.shape",self.dataset_A_mean.shape)
                                # print("\nself.dataset_B_mean.shape",self.dataset_B_mean.shape)
                                

                                # dataset_A_mean_dtw = self.dataset_A_mean[pathAq]
                                # dataset_A_std_dtw  = self.dataset_A_std[pathAq]
                                dataset_A_mean_dtw = self.test_dataset_A_mean
                                dataset_A_std_dtw  = self.test_dataset_A_std
                                
                                #dataset_B_mean_dtw = self.dataset_B_mean[pathBx] 
                                #dataset_B_std_dtw  = self.dataset_B_std[pathBq]
                                dataset_B_mean_dtw = self.test_dataset_B_mean 
                                dataset_B_std_dtw  = self.test_dataset_B_std
                                

                                #Generating Time Aligned Speech Files
                                # real_wav_full_A_DTW = decode_melspectrogram(self.vocoder, real_mel_full_A0[0], dataset_A_mean_dtw, dataset_A_std_dtw)
                                # fake_wav_full_B_DTW = decode_melspectrogram(self.vocoder, fake_mel_full_B0[0], dataset_B_mean_dtw, dataset_B_std_dtw)
                                
                                
                                import numpy as np
                                from scipy.signal import fftconvolve, hann
                                
                                def overlap_add(frames, hop_size):
                                    frame_length = frames.shape[1]
                                    num_frames = frames.shape[0]
                                    output_length = (num_frames - 1) * hop_size + frame_length

                                    output = np.zeros(output_length)
                                    for i in range(num_frames):
                                        start = i * hop_size
                                        end = start + frame_length
                                        output[start:end] += frames[i]

                                    return output

                                def overlap_add_win(frames, hop_size, window):
                                    frame_length = frames.shape[1]
                                    num_frames = frames.shape[0]
                                    output_length = (num_frames - 1) * hop_size + frame_length
                                    output = np.zeros(output_length)
                                    for i in range(num_frames):
                                        start = i * hop_size
                                        end = start + frame_length
                                        output[start:end] += frames[i] * window

                                    return output


                                """
                                def dtw(frames1, frames2):
                                        # Perform frame-wise dynamic time warping (DTW) between two sets of frames
                                        # ...

                                    return aligned_frames1, aligned_frames2

                                # Example usage
                                # Assume frames1 and frames2 are the input frames for DTW
                                frames1 = np.random.rand(10, 100)  # Shape: (num_frames, frame_length)
                                frames2 = np.random.rand(8, 100)   # Shape: (num_frames, frame_length)

                                aligned_frames1, aligned_frames2 = dtw(frames1, frames2)
                                """
                                        
                                # Assume hop_size is the desired hop size for overlap-add                                           
                                
                                """
                                hop_size = 50

                                # Create a Hann window of the same length as the frames
                                #window = hann(frames1.shape[1])
                                window = hann(x1.shape[1])
                                
                                aligned_frames1 = x1
                                aligned_frames2 = y1

                                
                                output1 = overlap_add(aligned_frames1.cpu(), hop_size)
                                output2 = overlap_add(aligned_frames2.cpu(), hop_size)

                                print("Output 1 shape:", output1.shape)
                                print("Output 2 shape:", output2.shape)
                                
                         
                                # print("\nx2.shape",x2.shape)
                                # print("\ny2.shape",y2.shape)
                                """
                                x2 = torch.unsqueeze(x1.T , 0)
                                y2 = torch.unsqueeze(y1.T , 0)
                                
                                p2 = torch.unsqueeze(p1.T , 0)
                                q2 = torch.unsqueeze(q1.T , 0)
                                
                                real_wav_full_B_dtw = decode_melspectrogram(self.vocoder, x2[0].detach(
                                ).cpu(), dataset_B_mean_dtw, dataset_B_std_dtw).cpu()                                
                                fake_wav_full_B_dtw = decode_melspectrogram(self.vocoder, y2[0].detach(
                                ).cpu(), dataset_B_mean_dtw, dataset_B_std_dtw).cpu() #MB:02.08.28 changed to B_Mean

                                real_wav_full_A_dtw = decode_melspectrogram(self.vocoder, p2[0].detach(
                                ).cpu() , dataset_A_mean_dtw, dataset_A_std_dtw).cpu()
                                fake_wav_full_A_dtw = decode_melspectrogram(self.vocoder, q2[0].detach(
                                ).cpu(), dataset_A_mean_dtw, dataset_A_std_dtw).cpu()#MB:02.08.28 changed to A_Mean
                                
                                """
                                real_wav_full_B_dtw = decode_melspectrogram(self.vocoder, x.T[0].detach(
                                ).cpu(), dataset_B_mean_dtw, dataset_B_std_dtw).cpu()
                                fake_wav_full_A_dtw = decode_melspectrogram(self.vocoder, q.T[0].detach(
                                ).cpu(), dataset_A_mean_dtw, dataset_A_std_dtw).cpu()
                               """

                            # self.logger.log_audio(
                            #     real_wav_full_A.T, "real_speaker_A_audio", self.sample_rate)
                            # self.logger.log_audio(
                            #     fake_wav_full_A.T, "fake_speaker_A_audio", self.sample_rate)
                            # self.logger.log_audio(
                            #     real_wav_full_B.T, "real_speaker_B_audio", self.sample_rate)
                            # self.logger.log_audio(
                            #     fake_wav_full_B.T, "fake_speaker_B_audio", self.sample_rate)
                            
                            fs=22050
                            testRealAfilePath ='report_files/wavs/test/test_real_A_wav020517.wav'
                            testRealBfilePath ='report_files/wavs/test/test_real_B_wav020517.wav' 
                            testFakeBfilePath ='report_files/wavs/test/test_fake_B_wav020517.wav'
                            testFakeAfilePath ='report_files/wavs/test/test_fake_A_wav020517.wav'
                            #testCycAfilePath  = 'report_files/wavs/test_cyc_A_sound020517.wav' 
                            #testCycBfilePath  = 'report_files/wavs/test_cyc_B_sound020517.wav' 
                            ###################### REAL and Fake Audio Files ################

                            torchaudio.save(testRealBfilePath, real_wav_full_B ,sample_rate=fs)
                            torchaudio.save(testRealAfilePath, real_wav_full_A ,sample_rate=fs)
                            torchaudio.save(testFakeBfilePath, fake_wav_full_B ,sample_rate=fs)
                            torchaudio.save(testFakeAfilePath, fake_wav_full_A ,sample_rate=fs)
                            #duration = len(real_A_wav)/ fs
                            #time = np.arange(0,duration,1/sample_rate) #time vector 
                            #print("\n real_A_wav duration:",duration)     
                            
                            #MB:03.01.14
                            if self.logger.epoch % 450 == 0: 
                                # I = '_'+str(i)
                                # testRealAfilePath ='report_files/wavs/test/test_real_A_wav020517'+I+'.wav'
                                # testRealBfilePath ='report_files/wavs/test/test_real_B_wav020517'+I+'.wav' 
                                # testFakeBfilePath ='report_files/wavs/test/test_fake_B_wav020517'+I+'.wav'
                                # testFakeAfilePath ='report_files/wavs/test/test_fake_A_wav020517'+I+'.wav'
                                # torchaudio.save(testRealBfilePath, real_wav_full_B ,sample_rate=fs)
                                # torchaudio.save(testRealAfilePath, real_wav_full_A ,sample_rate=fs)
                                # torchaudio.save(testFakeBfilePath, fake_wav_full_B ,sample_rate=fs)
                                # torchaudio.save(testFakeAfilePath, fake_wav_full_A ,sample_rate=fs)
                                pass
                            
                            if Time_Align_TestFiles == True :
                                #Generating Audio Test Files with DTW 02.07.11
                                testRealAfilePath_dtw ='report_files/wavs/test/test_real_A_wav020711dtw.wav'
                                testRealBfilePath_dtw ='report_files/wavs/test/test_real_B_wav020711dtw.wav' 
                                testFakeBfilePath_dtw ='report_files/wavs/test/test_fake_B_wav020711dtw.wav'
                                testFakeAfilePath_dtw ='report_files/wavs/test/test_fake_A_wav020711dtw.wav'
                                torchaudio.save(testRealBfilePath_dtw, real_wav_full_B_dtw ,sample_rate=fs)
                                torchaudio.save(testRealAfilePath_dtw, real_wav_full_A_dtw ,sample_rate=fs)
                                torchaudio.save(testFakeBfilePath_dtw, fake_wav_full_B_dtw ,sample_rate=fs)
                                torchaudio.save(testFakeAfilePath_dtw, fake_wav_full_A_dtw ,sample_rate=fs)
                                #
                                #MB:03.01.14
                                if self.logger.epoch % 450 == 0: 
                                    # I = '_'+str(i)
                                    # testRealAfilePath_dtw ='report_files/wavs/test/test_real_A_wav020711dtw'+I+'.wav'
                                    # testRealBfilePath_dtw ='report_files/wavs/test/test_real_B_wav020711dtw'+I+'.wav' 
                                    # testFakeBfilePath_dtw ='report_files/wavs/test/test_fake_B_wav020711dtw'+I+'.wav'
                                    # testFakeAfilePath_dtw ='report_files/wavs/test/test_fake_A_wav020711dtw'+I+'.wav'
                                    # torchaudio.save(testRealBfilePath_dtw, real_wav_full_B_dtw ,sample_rate=fs)
                                    # torchaudio.save(testRealAfilePath_dtw, real_wav_full_A_dtw ,sample_rate=fs)
                                    # torchaudio.save(testFakeBfilePath_dtw, fake_wav_full_B_dtw ,sample_rate=fs)
                                    # torchaudio.save(testFakeAfilePath_dtw, fake_wav_full_A_dtw ,sample_rate=fs)
                                    #
                                    pass

                            real_wav_full_A = real_wav_full_A.squeeze()
                            fake_wav_full_A = fake_wav_full_A.squeeze()
                            real_wav_full_B = real_wav_full_B.squeeze()
                            fake_wav_full_B = fake_wav_full_B.squeeze()
                            #print(f'\n{real_wav_full_A}')
                            #print(f'\n{real_wav_full_B}')
                            
                            #real_wav_full_A_dtw = real_wav_full_A_dtw.squeeze()
                            #fake_wav_full_A_dtw = fake_wav_full_A_dtw.squeeze()
                            real_wav_full_B_dtw = real_wav_full_B_dtw.squeeze()
                            fake_wav_full_B_dtw = fake_wav_full_B_dtw.squeeze()
                            
                            if( Metric_Measure_Test_Active and Metric_Measure_Test_mcd ): 
                                #print("\nMetric_Measure_Test_Active & Metric_Measure_Test_mmcd:True")
                                # instance of MCD class
                                # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
                                #mcd_toolbox = Calculate_MCD(MCD_mode="plain")
                                #mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
                                # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
                                                                
                                mcd_value11 = mcd_toolbox_test.calculate_mcd(real_wav_full_B.numpy(),
                                                                               fake_wav_full_B.numpy() )
                                
                                # mcd_value22 = mcd_toolbox_test.calculate_mcd(real_wav_full_A.numpy(),
                                #                                                fake_wav_full_B.numpy() )
                            
                                # mcd_value33 = mcd_toolbox_test.calculate_mcd(real_wav_full_A.numpy(),
                                #                                              fake_wav_full_A.numpy() )
                                
                                #mcd_value11 = mcd_toolbox.calculate_mcd(testRealAfilePath,testFakeBfilePath )
                                #mcd_value22 = mcd_toolbox.calculate_mcd(testRealBfilePath,testFakeAfilePath )
                                #mcd_value33 = mcd_toolbox.calculate_mcd(testRealBfilePath, testCycBfilePath ) 
                                #mcd_value44 = mcd_toolbox.calculate_mcd(testRealAfilePath, testCycAfilePath )
                                ##################################################################
                                #print("\nmcd_value11:(GB-RA MCD-DTW)",mcd_value11)
                                #print("mcd_value22: (GB-RB MCD-DTW)",mcd_value22)

                                #MMCD_GTB_test=(maxpymcd_test - mcd_value11) / maxpymcd_test
                                #MMCD_GTBB_test=(maxpymcd_test - mcd_value22) / maxpymcd_test                       
                                #MMCD_GTA_test=(maxpymcd_test - mcd_value33) / maxpymcd_test                      
                                
                                MCD_FTB_test = mcd_value11
                                #realAfakeB_mcd_test = mcd_value22
                                
                                #MMCD_GTBB_test=(maxpymcd_test - mcd_value22) / maxpymcd_test
                                #MMCD_GTA_test=(maxpymcd_test - mcd_value33) / maxpymcd_test

                                #MMCD_CTB_test=(maxpymcd -mcd_value33) / maxpymcd  
                                #MMCD_CTA_test=(maxpymcd -mcd_value44) / maxpymcd 
                                # print("\nMMCD_GTB",MMCD_GTB)
                                # print("\nMMCD_GTA",MMCD_GTA)  
                                #mmcd_test_list.append(MMCD_GTB_test)
                                
                                #mmcd_test_list.append(MMCD_FTB_test)
                                mcd_test_list.append (MCD_FTB_test)
                                #realAfakeB_mcd_list.append(realAfakeB_mcd_test )
                                
                            #Measurements in Test of Generators      
                            if(Metric_Measure_Test_Active and Metric_Measure_Test_stoi ):
                                #print("\nMetric_Measure_active & Metric_Measure_stoi:True")
                                #print("Measuring STIO score by pystoi is activated")                       
                                
                                # EStoi_test  = 'False'
                                # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
                                """
                                length=len(real_wav_full_B)
                                score=[]
                                for i in range(length):
                                  wo,_=librosa.load(wav_org[2], sr=8000)
                                  ws,_=librosa.load(wav_synth[2],sr=8000)
                                  n=len(wo)-len(ws)
                                  if n>0:
                                    ws=np.hstack((ws,np.zeros(abs(n))))
                                  elif n<0:
                                    wo=np.hstack((wo,np.zeros(abs(n))))
                                  score.append(pesq(wo,ws,8000))
                                print(np.mean(score))  
                                """
                                #minimum_w = min(len(real_A_wav),len(fake_B_wav) )
                                
                                # minimum_w1 = min( len( real_wav_full_B),len(fake_wav_full_B_dtw) )
                                # real_wav_full_B1 = real_wav_full_B[:minimum_w1]
                                # fake_wav_full_B1 = fake_wav_full_B_dtw[:minimum_w1]
                                minimum_w1 = min( len( real_wav_full_A),len(fake_wav_full_B) )
                                                 
                                real_wav_full_A1 = real_wav_full_A[:minimum_w1]
                                fake_wav_full_B1 = fake_wav_full_B[:minimum_w1]
                                
                                # minimum_w2 = min( len( real_wav_full_A,len(fake_wav_full_B) )
                                # fake_wav_full_B2 = fake_wav_full_B[:minimum_w2]
                                # real_wav_full_A2 = real_wav_full_A[:minimum_w2]
                                
                                # realB_fakeB_stoi = stoi(real_wav_full_B1.numpy(),
                                #                         fake_wav_full_B1.numpy(),fs,extended=EStoi)
                                                 
                                
                                realB_fakeB_dtwStoi = stoi(real_wav_full_B_dtw.numpy(),
                                                     fake_wav_full_B_dtw.numpy(),fs,extended=EStoi_test )
                                
                                #print("\nrealB_fakeB_dtwStoi",realB_fakeB_dtwStoi)
                                
                                # realB_fakeA_stoi = stoi(real_wav_full_B1.numpy(),
                                #                         fake_wav_full_A1.numpy(),fs,extended=EStoi)
                                
                                # minimum_w1 = min( len( real_wav_full_A),len(fake_wav_full_A) )
                                # real_wav_full_A1 = real_wav_full_A[:minimum_w1]
                                # fake_wav_full_A1 = fake_wav_full_A[:minimum_w1]
                                
                                # minimum_w2 = min( len( real_wav_full_B),len(fake_wav_full_B) )
                                # fake_wav_full_B1 = fake_wav_full_B[:minimum_w2]
                                # real_wav_full_B1 = real_wav_full_B[:minimum_w2]
                                
                                # realA_fakeB_stoi = stoi(real_wav_full_A1.numpy(),
                                #                         fake_wav_full_B1.numpy(),fs,extended=EStoi_test)
                                
                                #realA_fakeA_stoi = stoi(real_wav_full_A1.numpy(),
                                #                        fake_wav_full_A1.numpy(),fs,extended=EStoi)
                                
                                
                                # print("realA_genB_stoi",realA_genB_stoi)
                                # print("realB_genA_stoi",realB_genA_stoi)
                                # print("realB_cycB_stoi",realB_cycB_stoi)
                                # print("realA_cycA_stoi",realA_cycA_stoi) # TODO delete
                                #stoi_test_list.append(realA_fakeB_stoi)
                                                               
                                
                                # with open("report_files/realB_fakeB_test_dtwStoi.txt", "a") as f11:
                                #     f11.write(str(realB_fakeB_dtwStoi)+"\n")
                                
                                #stoi_test_list.append(realB_fakeB_stoi)
                                stoi_test_list.append(realB_fakeB_dtwStoi)
                                #realA_fakeB_stoi_list.append(realA_fakeB_stoi)
                                 
                                
                            #Measurements in Test of Generator     
                            if ( Metric_Measure_Test_Active and Metric_Measure_Test_cosinesim ):
                                #print("\nMetric_Measure__Test Active & Metric_Measure_Test_cosinesim: True")                          #sr = 16000
                                
                                if (vcc2018 == True ):
                                    sr = 22050
                                    # model_gru.cuda()
                                    # #mel_genb = model_gru.melspec_from_file( genBfilePath).cuda()
                                    # mel_fakeb = model_gru.melspec_from_array(fake_wav_full_B , sr ).cuda()
                                    # #mel_gena = model_gru.melspec_from_file( genAfilePath).cuda()
                                    # mel_realb = model_gru.melspec_from_array(real_wav_full_B , sr ).cuda()

                                    """
                                    mel_fakea = model_gru.melspec_from_array( fake_A_wav.T )
                                    #mel_reab = model_gru.melspec_from_file(realBfilePath).cuda()

                                    #mel_reaa = model_gru.melspec_from_file(realAfilePath).cuda()
                                    mel_reala = model_gru.melspec_from_array( real_A_wav.T )

                                    #mel_cycb = model_gru.melspec_from_file( cycBfilePath).cuda()
                                    #mel_cyca = model_gru.melspec_from_file( cycAfilePath).cuda()
                                    """
                                    #model_gru.cuda()
                                    #print("mel1.type ",mel1.type  )
                                    #print("mel1.size() ",mel1.size()  )
                                    #embed1 = model(mel1[None].cuda().requires_grad_(True))
                                    #include [None] to add the batch dimension
                                    # embed_fakeb = model_gru(mel_fakeb[None].cuda() )
                                    # embed_realb = model_gru(mel_realb[None].cuda() )
                                    
                                    #Using resemblyzer
                                    
                                    # testRealBfilePath ='report_files/wavs/test/test_real_B_wav020517.wav' 
                                    # testRealAfilePath ='report_files/wavs/test/test_real_A_wav020517.wav'
                                    # testFakeBfilePath ='report_files/wavs/test/test_fake_B_wav020517.wav'
                                    # testFakeAfilePath ='report_files/wavs/test/test_fake_A_wav020517.wav'
                                    
                                    #resemb_encoder = VoiceEncoder()
                                    # embed = encoder.embed_utterance(wav)
                                    # np.set_printoptions(precision=3, suppress=True)
                                    # print(embed)

                                    wav_test_realB  = preprocess_wav(testRealBfilePath)
                                    #wav_realA  = preprocess_wav(testRealAfilePath)
                                    wav_test_fakeB   = preprocess_wav(testFakeBfilePath )
                                    #wav_genA   = preprocess_wav(testFakeBfilePath )
                                    
                                    embed_test_realB = torch.from_numpy(resemb_encoder.embed_utterance(wav_test_realB)).cuda()
                                    embed_test_fakeB = torch.from_numpy(resemb_encoder.embed_utterance(wav_test_fakeB )).cuda()
                                    
                                    # embed_test_realA = torch.from_numpy(encoder.embed_utterance(wav_realA))
                                    # embed_test_fakeA  = torch.from_numpy(encoder.embed_utterance(wav_genA ))
                                    # #embed_test_cycB  = torch.from_numpy(encoder.embed_utterance(wav_cycB ))
                                    # #embed_test_cycA  = torch.from_numpy(encoder.embed_utterance(wav_cycA ))
                                    # # np.set_printoptions(precision=3, suppress=True)
                                    # # print(embed)

                                    cos = torch.nn.CosineSimilarity(dim=0)
                                    
                                    cosinesim_fakeBtarg=cos(embed_test_fakeB, embed_test_realB ).cpu()
                                    #cosinesim_fakeBtarg=cos(embed_test_fakeB, evalTRGspkEmbed  ).cpu()
                                    #cosinesim_fakeBtarg=cos(embed_test_fakeB, evalTM1spkEmbed  )
                                    # print("\nself.logger.epoch:",self.logger.epoch)
                                    # print("iteration i: ",i)
                                    # print("\nTest cosinesim_fakeBtarg",cosinesim_fakeBtarg)
                                    #cosinesim_fakeAtarg=cos(embed_test_fakeA, embed_test_realA ) 
                                    #cosinesim_fakeAtarg=cos(embed_test_fakeA, evalSF3spkEmbed )                                      

                                if( CMU == True ) :
                                    sr = 16000
                                    model_convgru.cuda() 
                                    embed_fakeb = model_convgru( fake_wav_full_B[None].cuda() )
                                    embed_realb = model_convgru( real_wav_full_B[None].cuda() )
                        
                                    #embed_fakeb = model_convgru( testFakeBfilePath[None].cuda() )
                                    #embed_realb = model_convgru( testRealBfilePath[None].cuda() )
                               
                                #model_gru.cuda()
                                #print("mel1.type ",mel1.type  )
                                #print("mel1.size() ",mel1.size()  )
                                #embed1 = model(mel1[None].cuda().requires_grad_(True))
                                #include [None] to add the batch dimension
                               
                                #cos = torch.nn.CosineSimilarity(dim=1)
                                #cos = torch.nn.CosineSimilarity(dim=0)
                                #cosine_np=cos(embed_fakeb, embed_realb ).cpu().numpy()  
                                #cosine_np=cos( embed_fakeb , gamma_norm  ).cpu().numpy()  
                                #cosine_np=cos( embed_fakeb , embed_realb ).cpu().numpy()  
                                
                                #cosinesim_fakeBtarg = cosine_np
                                
                                cosinesim_test_list.append(cosinesim_fakeBtarg)
                                #print("\n\nTest cosinesim_fakeBtarg",cosinesim_fakeBtarg)

                                #cosinesim_fakeBtarg.cpu().numpy()
                                #cosinesim_realBbetta=cos(embed_realb, lamnda_norm ) 
                                #print('\ncosinesim_realBbetta',cosinesim_realBbetta)
                                #cosinesim_fakeBtarg=cos(embed_fakeb, lamnda_norm )               
                                # print("\ncosinesim_genBtarg",cosinesim_genBtarg) 
                                # print("\ncosinesim_genAtarg",cosinesim_genAtarg)
                                # print("\ncosinesim_genBsorc",cosinesim_genBsorc)
                                # print("\ncosinesim_genAsorc",cosinesim_genAsorc)
                                
                            if ( Metric_Measure_Test_Active and Metric_Measure_Test_nisqa ): 
                                #print("\nMetric_Measure_Active & Metric_Measure_nisqa: True")
                                
                                Path ='report_files/wavs/test/' 
                
                                # Nisqatts_GB_norm = run_predict.nisqa_score(path + 'test_fake_B_wav020517.wav')/5.0
                                # Nisqatts_GA_norm = run_predict.nisqa_score(path + 'test_fake_A_wav020517.wav')/5.0 
                                
                                #Nisqatts_RB_norm = run_predict.nisqa_score(path + 'test_real_B_sound020517.wav')/5.0
                                #Nisqatts_RA_norm = run_predict.nisqa_score(path + 'test_real_A_sound020517.wav')/5.0
                                                                  
                                #Nisqatts_CB_norm = run_predict.nisqa_score(path + 'test_cyc_B_sound020517.wav')/5.0     
                                #Nisqatts_CA_norm = run_predict.nisqa_score(path + 'test_cyc_A_sound020517.wav')/5.0
                                #Nisqatts_CA_norm = run_predict.nisqa_score(path + cycAfilePath )/ 5.0 
                                #cycAfilePath='report_files/wavs/cyc_A_sound020430.wav'

                                # print("\nNisqatts_GB_norm",Nisqatts_GB_norm)
                                # print("\nNisqatts_GA_norm",Nisqatts_GA_norm)
                                # print("\nNisqatts_RB_norm",Nisqatts_RB_norm)
                                # print("\nNisqatts_RA_norm",Nisqatts_RA_norm) # TODO delete
                                
                                nisqa_test._loadDatasets()
                                mos_test = nisqa_test.predict().to_dict()

                                mos_test = dict( zip(mos_test['deg'].values(), mos_test['mos_pred'].values()) )
                                #print("\nmos_test",mos_test)
                                
                                Nisqatts_FB_norm = mos_test['test_fake_B_wav020517.wav'] / 5.0
                                
                                Nisqatts_FA_norm = mos_test['test_fake_A_wav020517.wav'] / 5.0
                                Nisqatts_RB_norm = mos_test['test_real_B_wav020517.wav'] / 5.0
                                Nisqatts_RA_norm = mos_test['test_real_A_wav020517.wav'] / 5.0
                                
                                # print("\n Nisqatts_FB_norm", Nisqatts_FB_norm )
                                # print("\n Nisqatts_FA_norm", Nisqatts_FA_norm )
                                # print("\n Nisqatts_RB_norm", Nisqatts_RB_norm )
                                # print("\n Nisqatts_RA_norm", Nisqatts_RA_norm ) 
                                
                                
                                nisqa_test_list.append(Nisqatts_FB_norm)           
                             
                            if ( Metric_Measure_Test_Active and Metric_Measure_Test_pesq ): 
                                #print("\nMetric_Measure_Active & Metric_Measure_pesq: True")
                                Path ='report_files/wavs/test/' 
                                from pypesq import pesq
                                
                                
                                """
                                length=len(real_wav_full_B)
                                score=[]
                                for i in range(length):
                                  wo,_=librosa.load(wav_org[2], sr=8000)
                                  ws,_=librosa.load(wav_synth[2],sr=8000)
                                  n=len(wo)-len(ws)
                                  if n>0:
                                    ws=np.hstack((ws,np.zeros(abs(n))))
                                  elif n<0:
                                    wo=np.hstack((wo,np.zeros(abs(n))))
                                  score.append(pesq(wo,ws,8000))
                                print(np.mean(score))  
                                """
                                
                                # Clean and den should have the same length, and be 1D
                                
                                #minimum_w = min(len(real_A_wav),len(fake_B_wav) )
                                minimum_w1 = min( len( real_wav_full_A),len(fake_wav_full_B) )
                                real_wav_full_A1 = real_wav_full_A[:minimum_w1]
                                fake_wav_full_B1 = fake_wav_full_B[:minimum_w1]
                                
                                minimum_w2 = min( len( real_wav_full_B),len(fake_wav_full_A) )
                                fake_wav_full_B2 = fake_wav_full_A[:minimum_w2]
                                real_wav_full_B2 = real_wav_full_B[:minimum_w2]

                                realAfakeB_test_pesqnorm = ( pesq( real_wav_full_A1.numpy() , fake_wav_full_B1.numpy() ) + 0.5) / 5.0
                                realBfakeB_test_pesqnorm = ( pesq( real_wav_full_B2.numpy() , fake_wav_full_B2.numpy() ) + 0.5) / 5.0

                                # print("PESQ Metric for test phase is Activated(Scores:")
                                # print("\nrealAfakeB_test_pesqnorm" ,realAfakeB_test_pesqnorm )
                                # print(  "realBfakeA_test_pesqnorm" ,realB_fakeA_test_pesqnorm )

                                pesq_test_list.append(realAfakeB_test_pesqnorm)
                                
                            if ( Metric_Measure_Test_Active and Metric_Measure_Test_SDR ): 
                                #print("\nMetric_Measure_Active & Metric_Measure_SDR: True")
                                Path ='report_files/wavs/test/' 
                                
                                # Clean and den should have the same length, and be 1D

                                #minimum_w = min(len(real_A_wav),len(fake_B_wav) )
                                minimum_w1 = min( len( real_wav_full_A),len(fake_wav_full_B) )
                                real_wav_full_A1 = real_wav_full_A[:minimum_w1]
                                fake_wav_full_B1 = fake_wav_full_B[:minimum_w1]
                                #print("\nminimum_w1", minimum_w1 )
                                
                                minimum_w2 = min( len( real_wav_full_B),len(fake_wav_full_B) )
                                fake_wav_full_A2 = fake_wav_full_B[:minimum_w2]
                                real_wav_full_B2 = real_wav_full_B[:minimum_w2]
                                #print("\nminimum_w2" ,minimum_w2)
                                
                                minimum_w3 = min( len( real_wav_full_B),len(fake_wav_full_B) )
                                fake_wav_full_B3 = fake_wav_full_B[:minimum_w3]
                                real_wav_full_B3 = real_wav_full_B[:minimum_w3]
                                # print("\nminimum_w3" ,minimum_w3)
                                
                                #from torchmetrics.audio import SignalDistortionRatio
                                #g = torch.manual_seed(1)
                                #preds = torch.randn(8000)
                                #target = torch.randn(8000)

                                sdr_db = SignalDistortionRatio()
                                #sdr_db = sdr(preds, target)
                                #sdr_mag = 10 ** (sdr_db / 10)

                                realAfakeB_test_sdrnorm = 1.8 * 10 ** ( sdr_db( real_wav_full_A1 , fake_wav_full_B1 ) / 10.0 ) 
                                realBfakeB_test_sdrnorm = 1.8 * 10 ** ( sdr_db( real_wav_full_B3 , fake_wav_full_B3 ) / 10.0 ) 
                                #realBfakeA_test_sdrnorm = 1.8 * 10 ** ( sdr_db( real_wav_full_B3 , fake_wav_full_A3 ) / 10.0 ) 

                                # print("SDR Metric for test phase is Activated(Scores:")
                                print("\nrealAfakeB_test_sdrnorm" , realAfakeB_test_sdrnorm )
                                
                                print("\nrealBfakeB_test_sdrnorm" , realBfakeB_test_sdrnorm )
                                # print("\nrealBfakeA_test_sdrnorm" ,realBfakeA_test_sdrnorm )
                                
                                sdr_test_list.append(realAfakeB_test_sdrnorm)
                                
                        d_loss_list.append(d_loss_full.item())
                        g_loss_list.append(g_loss_full.item())
                
                import numpy as np
                self.logger._log_scalars(
                        scalar_dict={'A.1_g_lossTest': np.mean(np.array(g_loss_list)) ,
                                     'A.2_d_lossTest': np.mean(np.array(d_loss_list))
                                    },
                        print_to_stdout = False)
                        
                if ( Metric_Measure_Test_Active ):
                    if ( Metric_Measure_Test_mcd ):
                        mcd_test_mean      = np.mean( np.array(mcd_test_list)       )
                        #mcd_test_realAfakeBmean = np.mean( np.array(realAfakeB_mcd_list)       )
                        
                    # if ( Metric_Measure_Test_mcd ):
                    #     mcd_test_mean      = np.mean( np.array(mcd_test_list)       )
                    
                    if ( Metric_Measure_Test_stoi ):
                        stoi_test_mean      = np.mean( np.array(stoi_test_list)       )
                        #stoi_test_realAfakeBmean=np.mean(np.array(realA_fakeB_stoi_list) )
                                                 
                    if ( Metric_Measure_Test_cosinesim ):
                        cosinesim_test_mean = np.mean( np.array(cosinesim_test_list)  )
                    if ( Metric_Measure_Test_nisqa ):
                        nisqa_test_mean     = np.mean( np.array(nisqa_test_list)      )
                                           
                    if ( Metric_Measure_Test_pesq ):
                        pesq_test_mean     = np.mean( np.array(pesq_test_list)        )
                    if ( Metric_Measure_Test_SDR ):
                        sdr_test_mean      =  np.mean( np.array(sdr_test_list)        )
                    
                    
                    
                # self.logger._log_scalars(
                #         scalar_dict={'B.test_1.MMCD': mmcd_test_mean,
                #                      'B.test_2.STOI': stoi_test_mean, 
                #                      'B.test_3.CosineSim':cosinesim_test_mean,
                #                      'B.test_4.NISQA': nisqa_test_mean
                #                     },
                #          print_to_stdout = False)
                
                # self.logger._log_scalars(
                #         scalar_dict={'B.test_1.MMCD': mmcd_test_mean,
                #                      'B.test_2.STOI': stoi_test_mean, 
                #                     },
                #          print_to_stdout = False)
                
                
                if ( Metric_Measure_Test_Active ):
                    scalar_dict = {} # TEST CODE
                    if Metric_Measure_Test_mcd:
                        scalar_dict['B.test_1.MCD']      = mcd_test_mean
                        #scalar_dict['B.test_1.1.MCDrAfB'] = mcd_test_realAfakeBmean
                        
                    # if Metric_Measure_Test_mcd:
                    #     scalar_dict['B.test_1.MCD']      = mcd_test_mean
                    
                    if Metric_Measure_Test_stoi:
                        scalar_dict['B.test_2.STOI']      = stoi_test_mean
                        #scalar_dict['B.test_2.1.STOIrAfB']= stoi_test_realAfakeBmean 
                        
                    if Metric_Measure_Test_cosinesim:
                        scalar_dict['B.test_3.CosineSim'] = cosinesim_test_mean
                    if Metric_Measure_Test_nisqa:
                        scalar_dict['B.test_4.NISQA']     = nisqa_test_mean 
                    if Metric_Measure_Test_pesq:
                        scalar_dict['B.test_5.PESQ']      = pesq_test_mean 
                    if Metric_Measure_Test_SDR:
                        scalar_dict['B.test_6.SDR' ]      =  sdr_test_mean 
                        
                    
                    self.logger._log_scalars(scalar_dict , print_to_stdout = True )

            self.logger.end_epoch()

if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    cycleGAN = MaskCycleGANVCTraining(args)
    cycleGAN.train()
