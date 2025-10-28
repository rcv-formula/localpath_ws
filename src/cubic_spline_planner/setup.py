from setuptools import setup
package_name = 'cubic_spline_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=['cubic_spline_planner'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/cubic_spline_planner']),
        ('share/cubic_spline_planner', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Auto-generated ROS2 package from localplanner.zip',
    license='Apache-2.0',
    tests_require=['pytest'],
    
)
