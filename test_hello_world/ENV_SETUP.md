# Setting Environment Variables on Reachy Mini

When the app runs through the Reachy Mini app framework, it doesn't inherit environment variables from your SSH session. Here are the ways to set them:

## Option 1: System-wide Environment Variables (Recommended)

Set environment variables system-wide so they're available to all services:

### On the Robot (SSH into it):

```bash
# Edit the environment file
sudo nano /etc/environment

# Add these lines:
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=eu-west-1
AWS_S3_BUCKET=your-s3-bucket-name
AWS_S3_KEY_PREFIX=transcribe/
```

**Important:** After editing `/etc/environment`, you need to restart the system or the service for changes to take effect:

```bash
sudo reboot
# OR restart the app service (if you know the service name)
```

## Option 2: User Profile (if app runs as specific user)

If the app runs as a specific user (e.g., `reachy`), add to that user's profile:

```bash
# SSH as the user the app runs as
ssh reachy@reachy-mini.local

# Edit the profile
nano ~/.bashrc
# OR
nano ~/.profile

# Add these lines:
export AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
export AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
export AWS_REGION=eu-west-1
export AWS_S3_BUCKET=your-s3-bucket-name
export AWS_S3_KEY_PREFIX=transcribe/

# Reload the profile
source ~/.bashrc
```

## Option 3: Systemd Service Override (if app runs as systemd service)

If the app runs as a systemd service, create an override file:

```bash
# Find the service name (might be something like reachy-mini-apps or test-hello-world)
systemctl list-units | grep reachy

# Create override directory
sudo mkdir -p /etc/systemd/system/reachy-mini-apps.service.d/

# Create override file
sudo nano /etc/systemd/system/reachy-mini-apps.service.d/override.conf

# Add:
[Service]
Environment="AWS_ACCESS_KEY_ID=your_aws_access_key_id_here"
Environment="AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here"
Environment="AWS_REGION=eu-west-1"
Environment="AWS_S3_BUCKET=your-s3-bucket-name"
Environment="AWS_S3_KEY_PREFIX=transcribe/"

# Reload systemd and restart service
sudo systemctl daemon-reload
sudo systemctl restart reachy-mini-apps.service
```

## Option 4: Check How the App is Started

To find out how the app is started, check:

```bash
# Check if it's a systemd service
systemctl status | grep reachy
systemctl status | grep test-hello

# Check running processes
ps aux | grep test-hello-world
ps aux | grep reachy

# Check for app configuration
ls -la /etc/reachy* 2>/dev/null
ls -la ~/.config/reachy* 2>/dev/null
```

## Verification

After setting environment variables, verify they're available:

```bash
# In a new SSH session, check if variables are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Test from Python
python3 -c "import os; print(os.getenv('AWS_ACCESS_KEY_ID'))"
```

## Troubleshooting

If environment variables still don't work:

1. **Check the app logs** to see what's happening:
   ```bash
   tail -f /tmp/test_hello_world.log
   ```

2. **Test AWS credentials manually**:
   ```bash
   python3 -c "import os, boto3; print(boto3.client('polly', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), region_name='eu-west-1').describe_voices())"
   ```

3. **Check if the app is reading from a different location** - the app might be looking for credentials in AWS credentials file:
   ```bash
   # Check for AWS credentials file
   ls -la ~/.aws/credentials
   ```

## Alternative: Use AWS Credentials File

Instead of environment variables, you can use AWS credentials file:

```bash
# Create AWS credentials directory
mkdir -p ~/.aws

# Create credentials file
nano ~/.aws/credentials

# Add:
[default]
aws_access_key_id = your_aws_access_key_id_here
aws_secret_access_key = your_aws_secret_access_key_here

# Create config file
nano ~/.aws/config

# Add:
[default]
region = eu-west-1
```

Then update `utils.py` to use boto3's default credential chain (which will check the credentials file automatically).

