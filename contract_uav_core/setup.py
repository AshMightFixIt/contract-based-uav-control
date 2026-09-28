"""contract_uav_core: the plain-Python core of the contract-based UAV controller.

Two ways to install it:
  pip install -e ./contract_uav_core                  (from the repository root)
  colcon build --packages-select contract_uav_core    (ament_python)

The data files register the package in the ament index for colcon.
"""
from setuptools import find_packages, setup

package_name = 'contract_uav_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    python_requires='>=3.7',
    install_requires=['numpy>=1.20.0'],
    extras_require={
        # Plotting in the top-level scripts and the pygame live visualizer (viz/).
        'viz': ['matplotlib>=3.3.0', 'pygame>=2.0.0'],
    },
    zip_safe=True,
    maintainer='Aswatth Sunil',
    maintainer_email='aswatths@umich.edu',
    description='Contract-based three-tier (PID/MPC/H-inf) adaptive UAV controller core: '
                'A/G contracts, contract-aware EKF, CBF safety filter, flight-mode supervisor.',
    license='MIT',
)
