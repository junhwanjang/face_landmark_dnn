import numpy as np


# import cv2

class LandmarkHelper(object):
    """
    Helper for different landmark type
    """

    @classmethod
    def parse(cls, line, landmark_type):
        """
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input text
            landmark_type: len of landmarks
        Return:
             see child parse
        Raises:
            unsupport type
        """
        if landmark_type == 5:
            return cls.__landmark5_txt_parse(line)
        elif landmark_type == 68:
            return cls.__landmark68_txt_parse(line)
        elif landmark_type == 83:
            return cls.__landmark83_txt_parse(line)
        else:
            raise Exception("Unsupport lanmark type...")

    @staticmethod
    def flip(a, landmark_type):
        """
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: len of landmarks
        Return:
             landmarks: new landmarks
        Raises:
              unsupport type
        """
        if landmark_type == 5:
            landmarks = np.concatenate((a[1, :], a[0, :], a[2, :], a[4, :], a[3, :]), axis=0)
        elif landmark_type == 68:
            landmarks = np.concatenate((
                # Outer face
                a[10:16][::-1], a[8:9], a[0:8][::-1],
                a[22:26][::-1], a[17:21][::-1],
                # Nose
                a[27:30], a[34:35][::-1], a[33:34], a[31:32][::-1],
                # Eyes
                a[42:47][::-1], a[36:41][::-1],
                # Mouth
                a[52:54][::-1], a[51:52], a[48:50][::-1],
                a[58:59][::-1], a[57:58], a[55:56][::-1],
                a[63:64][::-1], a[62:63], a[60:61][::-1],
                a[65:66][::-1], a[66:67], a[67:68][::-1]), axis=0)

        elif landmark_type == 83:
            landmarks = np.concatenate((
                a[10:19][::-1], a[9:10], a[0:9][::-1],
                a[35:36], a[36:43][::-1], a[43:48][::-1],
                a[48:51][::-1], a[19:20], a[20:27][::-1],
                a[27:32][::-1], a[32:35][::-1],
                a[56:60][::-1], a[55:56], a[51:55][::-1],
                a[60:61], a[61:72][::-1], a[72:73],
                a[73:78][::-1], a[80:81], a[81:82],
                a[78:79], a[79:80], a[82:83]), axis=0)
        else:
            raise Exception("Unsupport landmark type...")
        return landmarks.reshape([-1, 2])

    @staticmethod
    def get_scales(landmark_type):
        """
        use for scales bbox according to bbox of landmarks
        Args:
            landmark_type: len of landmarks
        Return:
            (min, max), min crop
        Raises:
            unsupport type
        """
        if landmark_type == 5:
            return (2.7, 3.3), 4.5
        elif landmark_type == 68:
            return (1.2, 1.5), 2.6
        elif landmark_type == 83:
            return (1.2, 1.5), 2.6
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def __landmark5_txt_parse(line):
        """
        Args:
            line: 0=file path, 1=[0:4] is bbox and [4:] is landmarks
        Return:
             file path and landmarks with numpy type
        """
        a = line.split()
        data = map(int, a[1:])
        pts = data[4:]  # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark68_txt_parse(line):
        """
        Args:
            line: 0=file path, 1=landmarks
        Return:
             file path and landmarks with numpy type
        Raises:
            No
        """
        if "version" in line or "points" in line or "{" in line or "}" in line:
            pass
        else:
            loc_x, loc_y = line.strip().split(sep=" ")
            print(loc_x, loc_y)
            # global info
        # return [loc_x, loc_y]

    @staticmethod
    def __landmark83_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks83, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        a1 = np.fromstring(a[1], dtype=int, count=166, sep=',')
        a1 = a1.reshape((-1, 2))
        return a[0], a1
