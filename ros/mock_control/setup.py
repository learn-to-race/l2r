from setuptools import setup

package_name = "mock_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yujunq",
    maintainer_email="12068945+yujunqin@users.noreply.github.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pose_pub = mock_control.pose_publisher:main",
            "img_pub = mock_control.img_publisher:main",
            "action_sub = mock_control.action_subscriber:main",
        ],
    },
)
