import glfw
import mujoco
import numpy as np
import matplotlib.pyplot as plt


def cubic_bezier_interpolation(z_start, z_end, t):
    t = np.clip(t, 0, 1)
    z_diff = z_end - z_start
    bezier = t ** 3 + 3 * (t ** 2 * (1 - t))
    return z_start + z_diff * bezier


def inverse_kinematics(foot_pos, foot_ids, data, model, tol=1e-9, max_iter=1000):
    for ind, foot_id in enumerate(foot_ids):
        target_pos = foot_pos[ind]
        error = None
        """Simple iterative IK solver using gradient descent."""
        for i in range(max_iter):  # Maximum iterations
            mujoco.mj_forward(model, data)  # Perform forward kinematics
            current_position = data.site_xpos[foot_id]
            error = target_pos - current_position
            if np.linalg.norm(error) < tol:
                break
            # Compute joint corrections using pseudo-inverse Jacobian
            jacobian_pos = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian_pos, None, foot_id)
            delta_q = np.linalg.pinv(jacobian_pos) @ error
            data.qpos += delta_q
        if np.linalg.norm(error) > tol:
            print(f"{foot_id}: not converged")
    return data.qpos

def generate_reference(v_x, v_y, w_z, foot_ids, default_foot_pos, data, model, swing_height=0.08, T=0.4, N=1000):
    q_ref = np.zeros((N, 12))
    foot_ref = np.zeros((N, 4, 3))
    default_foot_pos = np.array([default_foot_pos[foot_id] for foot_id in foot_ids])

    # First, integrate trajectory of CoM from mid-stance to mid-stance
    ts = np.linspace(0, T, N)
    theta = w_z * ts                 # Heading integrates simply
    if abs(w_z) < 0.01:
        x_t = v_x * ts
        y_t = v_y * ts
    else:
        x_t = (v_x * np.sin(w_z * ts) + v_y * (np.cos(w_z * ts) - 1)) / w_z
        y_t = (v_x * (1 - np.cos(w_z * ts)) + v_y * np.sin(w_z * ts)) / w_z

    p_com = np.stack((x_t, y_t, theta), axis=1)

    p_foot_0 = np.hstack((p_com[0, :2], np.array(0))) + (np.array([
            [ np.cos(p_com[0, 2]), np.sin(p_com[0, 2]), 0],
            [-np.sin(p_com[0, 2]), np.cos(p_com[0, 2]), 0],
            [0, 0, 1]
        ]) @ default_foot_pos.T).T

    p_foot_1 = np.hstack((p_com[-1, :2], np.array(0))) + (np.array([
            [ np.cos(p_com[-1, 2]), np.sin(p_com[-1, 2]), 0],
            [-np.sin(p_com[-1, 2]), np.cos(p_com[-1, 2]), 0],
            [0, 0, 1]
        ]) @ default_foot_pos.T).T

    # Next, design trajectory for each foot
    for i in range(N):
        t = ts[i]
        phase = np.clip((t / T - 0.25) * 2, 0, 1)
        # weird phasing. First 25 % of trajectory is last 50% of stance. Next 50% is swing. Final 25% is stance
        z = np.where(
            phase <= 0.5,
            cubic_bezier_interpolation(0, swing_height, 2 * phase),  # Upward portion of swing
            cubic_bezier_interpolation(swing_height, 0, 2 * phase - 1)  # downward portion of swing
        )
        # Spacial positions in global frame
        foot_pos = cubic_bezier_interpolation(p_foot_0, p_foot_1, phase)
        # Convert to body frame
        foot_pos = (np.array([
            [ np.cos(p_com[i, 2]), np.sin(p_com[i, 2]), 0],
            [-np.sin(p_com[i, 2]), np.cos(p_com[i, 2]), 0],
            [0, 0, 1]
        ]) @ (foot_pos - np.hstack((p_com[i, :2], np.array(0)))).T).T
        foot_pos[:, -1] += z

        foot_ref[i, :, :] = foot_pos
        q_ref[i, :] = inverse_kinematics(foot_pos, foot_ids, data, model)

    return ts, q_ref, foot_ref


if __name__ == "__main__":
    # Load the XML model
    model = mujoco.MjModel.from_xml_path("go2_fixed.xml")
    data = mujoco.MjData(model)

    # 1. Load keyframe and set the robot state
    keyframe_id = 0  # Assuming the keyframe ID you want to use is 0
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)  # Perform forward kinematics
    print("Loaded keyframe:", keyframe_id)

    foot_names = ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]
    base_foot_positions = {}
    foot_ids = []
    for foot in foot_names:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot)
        foot_ids.append(foot_id)
        site_pos = data.site_xpos[foot_id]
        base_foot_positions[foot_id] = site_pos.copy()
        print(f"Position of {foot}: {site_pos}")

    ts, q_ref, foot_ref = generate_reference(0, 0, 0, foot_ids, base_foot_positions, data, model)

    # Visualize foot trajectories
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        ax[i // 2, i % 2].plot(foot_ref[:, i, :])

    plt.figure()
    plt.plot(foot_ref[:, :, 0], foot_ref[:, :, 1])

    plt.show()

    # Initialize GLFW for visualization
    glfw.init()
    window = glfw.create_window(1200, 800, "MuJoCo Viewer", None, None)
    glfw.make_context_current(window)

    # Create visualization objects
    scene = mujoco.MjvScene(model, maxgeom=10000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    # Set camera properties
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Generate and visualize the inverse kinematics results
    for ind, t in enumerate(ts):
        data.qpos[:] = q_ref[ind]
        mujoco.mj_forward(model, data)

        # Update scene and render
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 1200, 800), scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    joint_names_isaac = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    ]
    joint_names_mujoco = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]

    plt.plot(q_ref)
    plt.show()
    mujoco_to_isaac = [joint_names_mujoco.index(joint_name) for joint_name in joint_names_isaac]
    isaac_to_mujoco = [joint_names_isaac.index(joint_name) for joint_name in joint_names_mujoco]
    np.savetxt("references/joint_trajectory.txt", q_ref[:, mujoco_to_isaac])
    plt.plot(q_ref[:, mujoco_to_isaac])
    plt.show()