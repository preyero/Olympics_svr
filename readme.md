# Running HF 🤗 LLMs on KMi's GPU server (through terminal)

### Initial checks

You need to be connected to OU network (oustaff) or use Citrix SSO to connect through the VPN.

### Connect to server

Use ssh to login into the remote server:

```commandline
$ ssh <your-oucu>@kmi-appsvr02.open.ac.uk
```

### Set up repo

Clone this repo into your user folder.

```commandline
$ git clone https://github.com/preyero/Olympics_svr.git
```

Activate conda environment.

```commandline
$ conda activate prl222_olympics
```

This conda environment has Python 3.8.12 and has installed the packages in requirements.txt

This environment was created as follows:

```commandline
$ conda create --name prl222_olympics python=3.12.8
$ pip install -r requirements.txt
```

### Check GPU status

These commands may be useful to use the GPU.

To check the GPU available:

```commandline
$ nvidia-smi
```

### Update parameters and execute using the bash file

You can call the Python script to run the experiment using 'run_get_video_llms.sh' .

Check the arguments are correct before execution. The file also exports terminal outputs to a txt file. 

To execute code using the GPU (there are two GPUs, can either use 0 or 1):

```commandline
$ tmux 
$ bash run_get_video_llms.sh
```

### Check results

The output will be a csv file (<input_name>_<model_id>.csv) with one column for each prompt. 

TODO: Pass also video file and add different types of context in the prompting function.
