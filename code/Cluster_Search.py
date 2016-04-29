# -*- coding: utf-8 -*-
# Setup the AWS Code
machine_image='ami-f44a869c'
instance_type='m1.small'
key_name='carpetri'
key_extension='.pem'
key_dir='~/.ssh'
login_user='ec2-user'
group_name='worldbank'
max_instance_count = 450
query_block_size=100

import os
import time
import boto
import boto.manage.cmdshell
import pandas as pd
from time import sleep

def launch_instance(ami='ami-7341831a',
                    instance_type='t1.micro',
                    key_name='jeff',
                    key_extension='.pem',
                    key_dir='~/.ssh',
                    group_name='worldbank',
                    ssh_port=22,
                    cidr='0.0.0.0/0',
                    tag='paws',
                    user_data=None,
                    cmd_shell=True,
                    login_user='ec2-user',
                    ssh_passwd=None,
                    wait_to_run=False):
    """
    Launch an instance and wait for it to start running.
    Returns a tuple consisting of the Instance object and the CmdShell
    object, if request, or None.

    ami        The ID of the Amazon Machine Image that this instance will
               be based on.  Default is a 64-bit Amazon Linux EBS image.

    instance_type The type of the instance.

    key_name   The name of the SSH Key used for logging into the instance.
               It will be created if it does not exist.

    key_extension The file extension for SSH private key files.
    
    key_dir    The path to the directory containing SSH private keys.
               This is usually ~/.ssh.

    group_name The name of the security group used to control access
               to the instance.  It will be created if it does not exist.

    ssh_port   The port number you want to use for SSH access (default 22).

    cidr       The CIDR block used to limit access to your instance.

    tag        A name that will be used to tag the instance so we can
               easily find it later.

    user_data  Data that will be passed to the newly started
               instance at launch and will be accessible via
               the metadata service running at http://169.254.169.254.

    cmd_shell  If true, a boto CmdShell object will be created and returned.
               This allows programmatic SSH access to the new instance.

    login_user The user name used when SSH'ing into new instance.  The
               default is 'ec2-user'

    ssh_passwd The password for your SSH key if it is encrypted with a
               passphrase.
    """
    cmd = None
    
    # Create a connection to EC2 service.
    # You can pass credentials in to the connect_ec2 method explicitly
    # or you can use the default credentials in your ~/.boto config file
    # as we are doing here.
    ec2 = boto.connect_ec2()

    # Check to see if specified keypair already exists.
    # If we get an InvalidKeyPair.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    try:
        key = ec2.get_all_key_pairs(keynames=[key_name])[0]
    except ec2.ResponseError, e:
        if e.code == 'InvalidKeyPair.NotFound':
            print 'Creating keypair: %s' % key_name
            # Create an SSH key to use when logging into instances.
            key = ec2.create_key_pair(key_name)
            
            # AWS will store the public key but the private key is
            # generated and returned and needs to be stored locally.
            # The save method will also chmod the file to protect
            # your private key.
            key.save(key_dir)
        else:
            raise

    # Check to see if specified security group already exists.
    # If we get an InvalidGroup.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    try:
        group = ec2.get_all_security_groups(groupnames=[group_name])[0]
    except ec2.ResponseError, e:
        if e.code == 'InvalidGroup.NotFound':
            print 'Creating Security Group: %s' % group_name
            # Create a security group to control access to instance via SSH.
            group = ec2.create_security_group(group_name,
                                              'A group that allows SSH access')
        else:
            raise

    # Add a rule to the security group to authorize SSH traffic
    # on the specified port.
    try:
        group.authorize('tcp', ssh_port, ssh_port, cidr)
    except ec2.ResponseError, e:
        if e.code == 'InvalidPermission.Duplicate':
            print 'Security Group: %s already authorized' % group_name
        else:
            raise

    # Now start up the instance.  The run_instances method
    # has many, many parameters but these are all we need
    # for now.
    reservation = ec2.run_instances(ami,
                                    key_name=key_name,
                                    security_groups=[group_name],
                                    instance_type=instance_type,
                                    user_data=user_data)

    # Find the actual Instance object inside the Reservation object
    # returned by EC2.

    instance = reservation.instances[0]

    # The instance has been launched but it's not yet up and
    # running.  Let's wait for it's state to change to 'running'.

    if wait_to_run:
        print 'waiting for instance'
        while instance.state != 'running':
            print '.'
            time.sleep(5)
            instance.update()
        print 'done'

    # Let's tag the instance with the specified label so we can
    # identify it later.
    instance.add_tag(tag)
    # The instance is now running, let's try to programmatically
    # SSH to the instance using Paramiko via boto CmdShell.
    #if cmd_shell:
    #    key_path = os.path.join(os.path.expanduser(key_dir),
    #                            key_name+key_extension)
    #    cmd = boto.manage.cmdshell.sshclient_from_instance(instance,
    #                                                      key_path,
    #                                                      user_name=login_user)
    #return (instance, cmd)
    return instance

# Define the Calculations to Be Run on Each Instance
startup_script = """#!/home/%(login_user)s/anaconda/bin/python
##### Get Data
print("Getting Data")
entity_block_start = %(entity_block_start)i
entity_block_stop = %(entity_block_stop)i #Inclusive
import pandas as pd
from pylab import *

entity_names = pd.DataFrame(%(entity_names)r,
                    columns=["Name"],
                    index=%(index)r) 
#pd.read_csv('../Data/Entities/all_entities.csv',index_col=0)

#### Establish parameters
error_method = "sleep"
max_attempt_count = 10

import sys
sys.path.append('/home/%(login_user)s/')
import google
from urllib2 import HTTPError

def perform_query(entity_name):
    query = google.search(entity_name, stop=1, only_standard=True, pause=0)
    return [url for url in query]
        
####### Build system for handling errors.
def handle_error(method="wait", sleep_time=5):
    if method=='wait':
        print "Waiting and trying again." 
        sleep(sleep_time)
    elif method=='tor':
        print "Reseting Tor node and cookies."
        reset_tor()
        
if error_method=='Tor': #NEEDS ROOT ACCESS TO RESET!
    import urllib2
    proxy = urllib2.ProxyHandler({'http': '127.0.0.1:8118'})
    opener = urllib2.build_opener(proxy)
    urllib2.install_opener(opener)
    from os import system
    #print urllib2.urlopen('http://icanhazip.com/').read()
    
    import subprocess

    def reset_tor():
        proc = subprocess.Popen(["ps aux | grep tor | grep -v grep"], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        target_process = out.split()[1]
        system("sudo kill -s SIGHUP %%s"%%target_process) #NEEDS ROOT ACCESS!

    reset_tor()
elif error_method=='sleep':
    from time import sleep


##### Initialize Data
query_results = pd.DataFrame(index=arange(entity_block_start,entity_block_stop+1), columns=["Name"]+map(str, range(10)))


##### Run queries
print("Running Queries")
for i in arange(entity_block_start,entity_block_stop+1):
    entity_name = entity_names.ix[i,"Name"]
    print i, entity_name

    attempt_count = 0
    while attempt_count<max_attempt_count:
        try:
            urls = perform_query(entity_name)
            break
        except HTTPError:
            print "HTTP Error."
            handle_error(method=error_method)
            google.cookie_jar.clear()
            attempt_count += 1
    else:
        raise Exception("Failed after %%i attempts to query entry %%i, name %%s"%%(attempt_count,i,entity_name))
            
    
    query_results.ix[i,"Name"] = entity_name
    n_urls = min(10,len(urls))
    query_results.ix[i,map(str, range(n_urls))] = urls[:n_urls]

#### Save results
print("Saving Results")
import csv
query_results.to_csv( "/home/%(login_user)s/all_entities_Google_results_ %(entity_block_start)i_to_%(entity_block_stop)i.csv", quoting=csv.QUOTE_ALL)
query_results.to_hdf( "/home/%(login_user)s/all_entities_Google_results_ %(entity_block_start)i_to_%(entity_block_stop)i.h5", 'df')
"""
# Load Entity Names and Define Blocks

entity_names = pd.read_csv('../Data/Entities/all_entities.csv',index_col=0)
entity_block_starts = entity_names.index[::query_block_size].tolist()
entity_block_stops = entity_names.index[query_block_size-1::query_block_size].tolist() + [entity_names.index[-1]]

# Create Query Commands and Send Out to New Instances
print("Sending Out Queries to New Instances")
from os import listdir
dirlist = listdir("../Search_Results/")
instances = []
instance_count = 0
for start, stop in zip(entity_block_starts, entity_block_stops):
    print start, stop
    
    if "all_entities_Google_results_%i_to_%i.csv"%(start,stop) in dirlist:
        print("Data already exists. Skipping.")
        continue #We've already done this condition, so don't requery it

    entities = entity_names.ix[start:stop,"Name"]
    parameters = {
    'index': entities.index.values,
    'entity_names': entities.values,
    'entity_block_start': start,
    'entity_block_stop': stop,
    'login_user': login_user
    }
    
    try:
        instance = launch_instance(user_data=startup_script%parameters,
                                        ami=machine_image,
                                        instance_type=instance_type,
                                        key_name=key_name,
                                        key_extension=key_extension,
                                        key_dir=key_extension,
                                        group_name=group_name,
                                        tag='world_bank_query')
        instance.add_tag("query_block_start",value=start)
        instance.add_tag("query_block_stop",value=stop)
        instances.append((instance))
        instance_count+=1
    except:
        pass
    if instance_count>max_instance_count:
        break


# Wait for Calculations to Run
print("Waiting for Calculations to Run")
sleep(900) #15 minutes

# Pull in Data and Terminate Instances
print("Pulling in Data and Terminating Instances")
key_path = os.path.join(os.path.expanduser(key_dir),
                            key_name+key_extension)

make_cmd = lambda instance:boto.manage.cmdshell.sshclient_from_instance(
    instance,key_path,user_name=login_user)

for instance in instances:
    print instance
    instance.update()
    if instance.state in ['shutting-down', 'terminated', 'pending']:
        print("Instance state is %s, skipping"%instance.state)
        continue
    else:
        start = int(instance.tags['query_block_start'])
        stop = int(instance.tags['query_block_stop'])
        print("Pulling query block %i to %i"%(start,stop))
        try:
            cmd = make_cmd(instance)
            cmd.get_file("all_entities_Google_results_%i_to_%i.csv" % (start,stop), "../Search_Results/all_entities_Google_results_ %i_to_%i.csv" % (start,stop))
            cmd.get_file("all_entities_Google_results_%i_to_%i.h5" % (start,stop), "../Search_Results/all_entities_Google_results_ %i_to_%i.h5" % (start,stop))
        except (IOError, AttributeError):
            print("Data not calculated for this instance.")
            pass
        instance.terminate()
