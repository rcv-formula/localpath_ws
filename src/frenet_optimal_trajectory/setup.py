from setuptools import setup
package_name = 'frenet_optimal_trajectory'

setup(
    name=package_name,
    version='0.1.0',
    packages=['frenet_optimal_trajectory'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/frenet_optimal_trajectory']),
        ('share/frenet_optimal_trajectory', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Auto-generated ROS2 package from localplanner.zip',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['frenet_planner_node = frenet_optimal_trajectory.frenet_planner_node:main']},
)
