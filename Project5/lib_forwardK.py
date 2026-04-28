#*-------------------------------------------------------------------------------------------------
#* Author: Sagar Ojha
#*         Roy Wu
#* Description: Forward Kinematics library
#* Note: Performs forward kinematics for CR3
#*-------------------------------------------------------------------------------------------------
import numpy as np
import math as ma
#*-------------------------------------------------------------------------------------------------
def XFormDH(a,al,d,th):
  """! Forms a homogeneous transformation matrix for given DH parameters
  @param a The offset between z-axes.
  @param al The angle of rotation around common normal.
  @param d The offset between x-axes.
  @param th The angle of rotation about z-axis.
  @return The homogeneous transformation.
  """
  al = al*ma.pi/180
  th = th*ma.pi/180
  m = np.array([[ma.cos(th),-ma.sin(th)*ma.cos(al),ma.sin(th)*ma.sin(al),a*ma.cos(th)],
                [ma.sin(th),ma.cos(th)*ma.cos(al),-ma.cos(th)*ma.sin(al),a*ma.sin(th)],
                [0,ma.sin(al),ma.cos(al),d],
                [0,0,0,1]])
  return m
#*-------------------------------------------------------------------------------------------------

#*-------------------------------------------------------------------------------------------------
def ht_compute(J):
  """! Computes the HT matrices for each frame w.r.t frame '0'.
  @param J The joint angle array.
  @return 6+3 homogeneous matrices w.r.t frame 0 (T01, T02...T09).
  """

  #* a, alpha, d, theta are the 4 parameters in the DH table 
  #* and they can be found in Sagar's UR23 paper
  #* https://ieeexplore.ieee.org/document/10202406
  a = np.array([0, 274, 230, 0, 0, 0, 0, 0, 0])/1000
  α = np.array([90, 0, 0, 90, 90, 0, 0, 0, 0])
  d = np.array([134.8, 0, 0, 128.3, 116, 105, 19.6, 178.3, 39])/1000
  θ = np.array([J[0], 90+J[1], J[2], 90+J[3], 180+J[4], J[5], 0, 0, 0])

  T01=XFormDH(a[0],α[0],d[0],θ[0])     #Frame 1 wrt 0
  print(f"T01: {T01}")
  T12=XFormDH(a[1],α[1],d[1],θ[1])     #Frame 2 wrt 1
  T23=XFormDH(a[2],α[2],d[2],θ[2])     #Frame 3 wrt 2
  T34=XFormDH(a[3],α[3],d[3],θ[3])     #Frame 4 wrt 3
  T45=XFormDH(a[4],α[4],d[4],θ[4])     #Frame 5 wrt 4
  T56=XFormDH(a[5],α[5],d[5],θ[5])     #Frame 6 wrt 5
  T67=XFormDH(a[6],α[6],d[6],θ[6])     #Frame 7 wrt 6
  T78=XFormDH(a[7],α[7],d[7],θ[7])     #Frame 8 wrt 7
  T89=XFormDH(a[8],α[8],d[8],θ[8])     #Frame 9 wrt 8

  #* One extra frame (frame 7) added at the blue mounting plate when the gripper is removed
  #* Two extra frames (8 and 9) are added to the RG2 gripper end when it is open and closed 
  #* The 3 extra frames do not have any rotational component at all
  #* 6+1+2=9 9 frames in total


  T09=T01@T12@T23@T34@T45@T56@T67@T78@T89
  T08=T01@T12@T23@T34@T45@T56@T67@T78
  T07=T01@T12@T23@T34@T45@T56@T67
  T06=T01@T12@T23@T34@T45@T56
  T05=T01@T12@T23@T34@T45
  T04=T01@T12@T23@T34
  T03=T01@T12@T23
  T02=T01@T12

  return T01, T02, T03, T04, T05, T06, T07, T08, T09


#*-------------------------------------------------------------------------------------------------

#*-------------------------------------------------------------------------------------------------
def forward_kinematics(J):
  """! Performs Forward Kinematics on CR3.
  """
  T01, T02, T03, T04, T05, T06, T07, T08, T09 = ht_compute(J)
  print(f'The joint angles are: {J}')
  print(f'The end-effector is at: {T09[:3, 3].round(4)}')
  print(f'Forward Kinematics END!\U0001f642')
  return T01, T02, T03, T04, T05, T06, T09
#*-------------------------------------------------------------------------------------------------