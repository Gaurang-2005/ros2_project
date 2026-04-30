from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pick_and_drop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[

        # Required
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # 🔥 ADD THESE
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.dae')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'gazebo'), glob('gazebo/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gaurang_gupta',
    maintainer_email='gaurang_gupta@todo.todo',
    description='Pick and place robotic arm',
    license='TODO',
    entry_points={
        'console_scripts': [
            'control = pick_and_drop.control:main',
        ],
    },
)