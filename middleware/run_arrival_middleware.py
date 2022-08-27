import threading
from middleware import ArrivalMiddleware

mid = ArrivalMiddleware()

action_proxy = threading.Thread(target=mid.proxy_action)
pose_proxy = threading.Thread(target=mid.proxy_pose)
cam_proxy = threading.Thread(target=mid.proxy_cam)

action_proxy.start()
pose_proxy.start()
cam_proxy.start()

action_proxy.join()
pose_proxy.join()
cam_proxy.join()