from setuptools import setup, find_packages

setup(
    name="soumyagpt",
    version="0.1.0",
    description="Voice-cloned CLI chatbot using Gemini API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Soumya",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/soumyagpt",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "faster-whisper>=1.0.0",
        "google-generativeai>=0.8.0",
        "pyttsx3>=2.90",
        "edge-tts>=6.1.0",
        "sounddevice>=0.4.0",
        "soundfile>=0.12.0",
        "numpy>=1.21.0",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "pydub>=0.25.0",
        "colorama>=0.4.0",
        "rich>=13.0.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "soumyagpt=soumyagpt.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)