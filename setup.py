from setuptools import setup, find_packages

setup(
    name="chemotaxis_high_res_autoscope",  # Replace "YourPackageName" with the name of your package
    version="0.1",  # Replace "0.1" with the current version of your package
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        # Example: 'numpy', 'pandas>=1.0',
    ],
    # Optional fields:
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the package",
    url="https://github.com/yourusername/yourpackagename",  # Replace with the URL of your project
    # More fields can be added as per your package's requirements
)
