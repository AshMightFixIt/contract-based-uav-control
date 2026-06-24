import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'contract_uav_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aswatth Sunil',
    maintainer_email='aswatths@umich.edu',
    description='Contract-based three-tier adaptive drone controller as a PX4 offboard ROS 2 node.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = contract_uav_control.controller_node:main',
        ],
    },
)
