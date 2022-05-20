import logging
import os
import sys


def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger



def vocab_opts(parser):

    parser.add_argument('-vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")

    parser.add_argument('-max_unk_words', type=int, default=1000,
                        help="Maximum number of unknown words the model supports (mainly for masking in loss)")


def preprocess_opts(parser):
    parser.add_argument('-data_dir', default = 'data/kp20k_separated', help='The source file of the data')
    parser.add_argument('-save_data_dir', default = 'data/kp20k_separated/Full', help='The saving path for the data')
    parser.add_argument('-remove_title_eos', default = True, help='Remove the eos after the title')
    parser.add_argument('-one2many', default = True, help='Save one2many file.')
    parser.add_argument('-log_path', type=str, default="logs")
    return parser


def model_opts(parser):
    parser.add_argument('-word_vec_size', type=int, default=512,
                        help='Word embedding for both.')

    parser.add_argument('-enc_layers', type=int, default=6,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=6,
                        help='Number of layers in the decoder')
    parser.add_argument('-dropout', type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument('-d_model', type=int, default=512,
                        help="Model dimension")
    parser.add_argument('-n_head', type=int, default=8,
                        help="Multi-head numbers")
    parser.add_argument('-dim_ff', type=int, default=2048,
                        help="Feed-forward dimension")

    parser.add_argument('-copy_attention', default=True,
                        help='Train the model with copy mechanism.')

    parser.add_argument('-max_kp_len', type=int, default=6,
                        help='the maximum length of keyphrase, this is aimed for easily '
                             'running multiple control codes in parallel')
    parser.add_argument('-max_kp_num', type=int, default=20,
                        help='the number of control codes')
    parser.add_argument('-fix_kp_num_len',  default=False,
                        help='fix the maximun kp length and number')
    parser.add_argument('-seperate_pre_ab', default=False,
                        help='Whether use a seperate set loss')
    parser.add_argument('-SCT', default=False,
                        help='fix the maximun kp length and number')
    parser.add_argument('-tripleloss', default=False,
                        help='Whether use a seperate set loss')
    parser.add_argument('-loss_indomain', default=True,
                        help='fix the maximun kp length and number')
    parser.add_argument('-loss_outdomain', default=True,
                        help='Whether use a seperate set loss')
    parser.add_argument('-RL', default=True,
                        help='Whether use a seperate set loss')


def train_opts(parser):
    parser.add_argument('-data',
                        help="""Path prefix to the "train.one2one.pt" and
                        "train.one2many.pt" file path from preprocess.py""")
    parser.add_argument('-vocab',
                        help="""Path prefix to the "vocab.pt"
                        file path from preprocess.py""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    parser.add_argument('-exp_path', type=str, default="exp",
                        help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="model",
                        help="Path of checkpoints.")

    parser.add_argument('-start_checkpoint_at', type=int, default=1,
                        help="""Start checkpointing every epoch after and including
                                this epoch""")
    parser.add_argument('-checkpoint_interval', type=int, default= 2000,
                        help='Run validation and save model parameters at this interval.')
    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-early_stop_tolerance', type=int, default=6,
                        help="Stop training if it doesn't improve any more for several rounds of validation")

    parser.add_argument('-gpuid', type=int,
                        help="Use CUDA on the selected device.")
    parser.add_argument('-seed', type=int, default=27,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=0,
                        help='Number of workers for generating batches')

    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-max_grad_norm', type=float, default=1,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-loss_normalization', default="tokens", choices=['tokens', 'batches'],
                        help="Normalize the cross-entropy loss by the number of tokens or batch size")

    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                            Recommended settings: sgd = 1, adagrad = 0.1,
                            adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                            this much if (i) perplexity does not decrease on the
                            validation set or (ii) epoch has gone past
                            start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                            epoch""")

    parser.add_argument('-one2many', default=True,
                        help='If true, it will not split a sample into multiple src-keyphrase pairs')

    parser.add_argument('-loss_scale', type=float, default=0.2,
                        help='scale loss of NULL_WORD, used in set mode and seperate_pre_ab is False')
    parser.add_argument('-loss_scale_pre', type=float, default=0.2,
                        help='scale loss of NULL_WORD for present set, used in set mode and seperate_pre_ab is True')
    parser.add_argument('-loss_scale_ab', type=float, default=0.1,
                        help='scale loss of NULL_WORD for absent set, used in set mode and seperate_pre_ab is True')
    parser.add_argument('-set_loss',  default=False,
                        help='Whether to use set loss')
    parser.add_argument('-assign_steps', type=int, default=2,
                        help='Assignment steps K')


def predict_opts(parser):
    parser.add_argument('-src_file', help="""Path to source file""")
    parser.add_argument('-remove_title_eos', action="store_true",
                        help='Remove the eos token at the end of title')

    parser.add_argument('-vocab',
                        help="""Path prefix to the "vocab.pt"
                            file path from preprocess.py""")
    parser.add_argument('-model',
                        help='Path to model .pt file')
    parser.add_argument('-pred_path', type=str, default="pred/%s.%s",
                        help="Path of outputs of predictions.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")

    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                            reproducibility.""")
    parser.add_argument('-batch_size', type=int, default=12,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=0,
                        help='Number of workers for generating batches')

    parser.add_argument('-beam_size', type=int, default=200,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=None,
                        help='Pick the top n_best sequences from beam_search, if n_best is None, then n_best=beam_size')
    parser.add_argument('-max_length', type=int, default=6,
                        help='Maximum prediction length.')

    parser.add_argument('-one2many', action="store_true",
                        help='If true, it will not split a sample into multiple src-keyphrase pairs')

    parser.add_argument('-length_penalty_factor', type=float, default=0.,
                        help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    parser.add_argument('-coverage_penalty_factor', type=float, default=0.,
                        help="""Coverage penalty parameter""")
    parser.add_argument('-length_penalty', default='none', choices=['none', 'avg'],
                        help="""Length Penalty to use.""")
    parser.add_argument('-coverage_penalty', default='none', choices=['none', 'summary'],
                        help="""Coverage Penalty to use.""")
    parser.add_argument('-block_ngram_repeat', type=int, default=0,
                        help='Block repeat of n-gram')
    parser.add_argument('-ignore_when_blocking', nargs='+', type=str,
                        default=['<sep>'],
                        help="""Ignore these strings when blocking repeats.
                                       You want to block sentence delimiters.""")

    parser.add_argument('-replace_unk', action="store_true",
                        help='Replace the unk token with the token of highest attention score.')


def post_predict_opts(parser):
    parser.add_argument('-pred_file_path', type=str,
                        help="Path of the prediction file.")
    parser.add_argument('-src_file_path', type=str,
                        help="Path of the source text file.")
    parser.add_argument('-trg_file_path', type=str,
                        help="Path of the target text file.")
    parser.add_argument('-export_filtered_pred', action="store_true",
                        help="Export the filtered predictions to a file or not")
    parser.add_argument('-filtered_pred_path', type=str, default="",
                        help="Path of the folder for storing the filtered prediction")
    parser.add_argument('-exp_path', type=str, default="",
                        help="Path of experiment log/plot.")
    parser.add_argument('-disable_extra_one_word_filter', action="store_true",
                        help="If False, it will only keep the first one-word prediction")
    parser.add_argument('-disable_valid_filter', action="store_true",
                        help="If False, it will remove all the invalid predictions")
    parser.add_argument('-num_preds', type=int, default=200,
                        help='It will only consider the first num_preds keyphrases in each line of the prediction file')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Print out the metric at each step or not')
    parser.add_argument('-match_by_str', action="store_true", default=False,
                        help='If false, match the words at word level when checking present keyphrase. Else, match the words at string level.')
    parser.add_argument('-invalidate_unk', action="store_true", default=False,
                        help='Treat unk as invalid output')
    parser.add_argument('-target_separated', action="store_true", default=False,
                        help='The targets has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-prediction_separated', action="store_true", default=False,
                        help='The predictions has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-reverse_sorting', action="store_true", default=False,
                        help='Only effective in target separated.')
    parser.add_argument('-tune_f1_v', action="store_true", default=False,
                        help='For tuning the F1@V score.')
    parser.add_argument('-all_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='only allow integer or M')
    parser.add_argument('-present_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='')
    parser.add_argument('-absent_ks', nargs='+', default=['5', '10', '50', 'M'], type=str,
                        help='')
    parser.add_argument('-target_already_stemmed', action="store_true", default=False,
                        help='If it is true, it will not stem the target keyphrases.')
    parser.add_argument('-meng_rui_precision', action="store_true", default=False,
                        help='If it is true, when computing precision, it will divided by the number pf predictions, instead of divided by k.')
    parser.add_argument('-use_name_variations', action="store_true", default=False,
                        help='Match the ground-truth with name variations.')
