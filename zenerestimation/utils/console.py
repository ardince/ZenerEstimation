"""
Console helper.
"""

class Console:

    WIDTH = 60

    @staticmethod
    def header(title):

        print("=" * Console.WIDTH)
        print("ZenerEstimation")
        print(title)
        print("=" * Console.WIDTH)
        print()

    @staticmethod
    def section(title):

        print()
        print(title)
        print("-" * len(title))

    @staticmethod
    def info(message):

        print(f"[INFO] {message}")

    @staticmethod
    def success(message):

        print(f"[ OK ] {message}")

    @staticmethod
    def warning(message):

        print(f"[WARN] {message}")

    @staticmethod
    def footer(version):

        print()
        print("=" * Console.WIDTH)
        print(f"Framework Version : {version}")
        print("Demo Completed Successfully")
        print("=" * Console.WIDTH)