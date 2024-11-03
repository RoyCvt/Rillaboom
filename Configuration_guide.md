# Steps to configure Rillaboom on an android tablet:

1) Install the **Andronix** app from the Play Store.

2) Install the **Termux** app from the Play Store.

3) Open the **Andronix** app and select **Linux Distribution**.

4) In the **Distro Selection** window choose **Ubuntu** and select version **20.04**.

5) In the **GUI Selection** window choose **Desktop Environment** and then the **XFCE** environment.

6) Open the **Termux** app and paste the command that was automatically copied to your clipboard.
> [!NOTE]
> If the command wasn't copied, go back to **Andronix** and click on the **Recopy Command** button before pasting in **Termux**.

7) A system window will open listing apps with **Access all Files** permission. choose **Termux** from the list and tap on the toggle switch to give the permission.

8) Close the permissions window to go back to the **Termux** terminal screen and procceed to cofiguring the time zone and keyboard settings:
- for the **Geographic area**, type **6** for **Asia** and press enter.
- for the **Time zone**, type **37** for **Jerusalem** and press enter.
- for the **Country of origin for the keyboard**, first press enter a couple of times to show all options. Then type **31** for **English (US)** and press enter.
- for the **Keyboard layout**, type **1** for **English (US)** and press enter.

9) Choose a new password for the **VNC Server** and then re-enter it for verification.

10) You will be asked if you want to enter a view-only password. type **n** and press enter.

11) Install Git:
> sudo apt install git
   
> [!NOTE]
> When asked if you want to continue type **Y** and press enter.  

12) Download the project to the Documents folder:
> cd Documents  
> git clone https://github.com/RoyCvt/Rillaboom.git  
> cd Rillaboom  

13) Install pip:
> sudo apt install python3-pip  

> [!NOTE]
> When asked if you want to continue type **Y** and press enter.  
   
14) Install venv:
> sudo apt install python3.8-venv  

15) Create a virtual environment with the required dependencies:
> python3 -m venv rillaboom-env  
> sudo chmod -x rillaboom-env/bin/activate  
> source rillaboom-env/bin/activate  

16) Install the required dependencies:
> pip install -r requirements.txt  

