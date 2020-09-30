
function print(msg) {
  console.log(msg)
}

//model functions
function inference(x_input, var_list, index)
{
  const dropout_rate = 0.2
  const non_dropout_scaledown = 1.0 - dropout_rate

  function dropout_mask(shape) {
    return tf.mul(tf.ones(shape), tf.tensor(non_dropout_scaledown))
  }

  function full_layer(input, var_list, index) {
    filter_weights = var_list[index + 0]
    filter_bias = var_list[index + 1]
    net_convo = tf.matMul(input, filter_weights)
    net = tf.add(net_convo, filter_bias)
    return net
  }

  var fc1 = tf.leakyRelu(full_layer(x_input, var_list, index))
  index += 2
  fc1 = tf.mul(dropout_mask(fc1.shape), fc1)

  var fc2 = tf.leakyRelu(full_layer(fc1, var_list, index))
  index += 2

  var fc3 = tf.leakyRelu(full_layer(fc2, var_list, index))
  index += 2
  var fc4 = tf.leakyRelu(full_layer(fc3, var_list, index))
  index += 2
  var fc5 = tf.leakyRelu(full_layer(fc4, var_list, index))
  index += 2

  var fc6 = tf.leakyRelu(full_layer(fc5, var_list, index))
  index += 2
  var fc7 = tf.leakyRelu(full_layer(fc6, var_list, index))
  index += 2

  var fc_final = full_layer(fc7, var_list, index)
  index += 2

  return fc_final
}
function decoder(z, var_list, index) {

  const final_image_height = 6
  const final_image_width = 8
  const final_image_depth = 256
  depth_in = final_image_depth
  in_width = final_image_width
  in_height = final_image_height

  function convo_transpose_layer(input_t, depth_out, var_list, index, window_stride)
  {
    in_width *= window_stride
    in_height *= window_stride
    var filter_weights = var_list[index + 0]
    var filter_bias = var_list[index + 1]
    var net_convo = tf.conv2dTranspose(input_t, filter_weights, [input_t.shape[0], in_height, in_width, depth_out], 
      [window_stride, window_stride], 'same')
    var net = tf.add(net_convo, filter_bias)
    depth_in = depth_out
    return net
  }

  function full_layer(input_t, var_list, index){
    filter_weights = var_list[index + 0]
    filter_bias = var_list[index + 1]
    net_convo = tf.matMul(input_t, filter_weights)
    net = tf.add(net_convo, filter_bias)
    return net
  }

  var fc = tf.leakyRelu(full_layer(z, var_list, index))
  index += 2
  var reform_fc = tf.reshape(fc, [-1, final_image_height, final_image_width, final_image_depth])

  var conv1 = tf.leakyRelu(convo_transpose_layer(reform_fc, 256, var_list, index, 1))
  index += 2
  var conv2 = tf.leakyRelu(convo_transpose_layer(conv1, 128, var_list, index, 2))
  index += 2
  var conv3 = tf.leakyRelu(convo_transpose_layer(conv2, 128, var_list, index, 1))
  index += 2
  var conv4 = tf.leakyRelu(convo_transpose_layer(conv3, 64, var_list, index, 2))
  index += 2
  var conv5 = tf.leakyRelu(convo_transpose_layer(conv4, 64, var_list, index, 1))
  index += 2
  var conv6 = tf.tanh(convo_transpose_layer(conv5, 3, var_list, index, 2))
  index += 2
  //conv2.print()

  return conv6
}
function reverse_parse(images) {
  
  const x = tf.mul(tf.div(tf.add(images, tf.tensor(1)), tf.tensor(2)), tf.tensor(255.0))
  return tf.cast(tf.maximum(tf.zeros(images.shape), 
                            tf.minimum(tf.mul(tf.ones(images.shape), tf.tensor(255.0)), x)), 'int32')
}
async function download_weights()
{
  var model_json = await fetch('saved_model.json')
  model_json = await model_json.json()
  trackers_arr = model_json.trackers


  var model_data = await fetch('saved_model.bin')
  model_data = await model_data.arrayBuffer()
  model_data = new Float32Array(model_data)
  model_data = Array.prototype.slice.call(model_data)
  //model_data = model_data.slice(2, 5) //gives indexes 2,3,4

  var var_list = []
  var data_index = 0
  function load_var(shape) {
    shape_len = 1
    for (var i = 0; i < shape.length; i++) {
      shape_len *= shape[i]
    }
    var start_index = data_index
    var end_index = data_index + shape_len
    data_index += shape_len

    var var_data = model_data.slice(start_index, end_index)
    var var_data_t = tf.tensor(var_data)
    var_data_t = tf.keep(tf.reshape(var_data_t, shape))
    var_list.push(var_data_t)
  }

  for (var latent_i = 0; latent_i < model_json.trackers.length; latent_i++) {
    for (var var_i = 0; var_i < model_json.latent.length; var_i++) {
      var shape = model_json.latent[var_i]
      load_var(shape)
    }
  }
  for (var i = 0; i < model_json.decoder.length; i++) {
    load_var(model_json.decoder[i])
  }
  if (data_index != model_data.length)
    print('Warining: Unit test fail')
  return [model_json, var_list, trackers_arr]
}
//drawing functions
var camera_canvas
var camera_ctx
var canvas
var ctx
var rightPressed = false
var leftPressed = false
var upPressed = false
var downPressed = false

document.addEventListener('keydown', keyHandler, false)
document.addEventListener('keyup', keyHandler, false)


function keyHandler(e) {
  e.preventDefault();
  if (e.key == 'Right' || e.key == 'ArrowRight' || e.key == 'd') {
    rightPressed = (e.type == 'keydown')
  }
  else if (e.key == 'Left' || e.key == 'ArrowLeft' || e.key == 'a') {
    leftPressed = (e.type == 'keydown')
  }
  else if (e.key == 'Up' || e.key == 'ArrowUp' || e.key == 'w') {
    upPressed = (e.type == 'keydown')
  }
  else if (e.key == 'Down' || e.key == 'ArrowDown' || e.key == 's') {
    downPressed = (e.type == 'keydown')
  }
}


const BOARD_RANGE = 0.8  // workable board range.Max is 1 for [-1, 1]


const BOARD_START_X = 50
const BOARD_START_Y = 50
const BOARD_END_X = 350
const BOARD_END_Y = 350

function to_board_x(x) {
  return BOARD_START_X + to_board_x_length(x + 1)
}
function to_board_y(y) {
  return BOARD_START_Y + to_board_y_length(1 - y)
}
function to_board_x_length(x) {
  return (BOARD_END_X - BOARD_START_X) * x / 2
}
function to_board_y_length(y) {
  return (BOARD_END_Y - BOARD_START_Y) * y / 2
}
function to_board(x,y)
{
  return [to_board_x(x), to_board_y(y)]
}
function to_board_rect(x,y,w,h)
{
  return [to_board_x(x), to_board_y(y), to_board_x_length(w), to_board_y_length(h)]
}

var center_x = 0
var center_y = 0
var bearing = 45
var selected_target_index = 0
var trackers_arr;
var camera_image;
const CAMERA_HEIGHT = 480
const CAMERA_WIDTH = 640
const MAX_LATENT = 2.0
var update_camera_image_on_next_tick = true
var load_slider_values_on_next_tick = false
var sliders = []
var last_ten_times = []
var speed_scale = 1
var warning_up = false
var overlay = true
var overlay_text_image
var overlay_alpha = -1
var overlay_alpha_increasing = true
var overlay_image 

function draw(){
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  ctx.beginPath()
  var a = to_board_rect(-1, 1, 2, 2)
  ctx.rect(a[0], a[1], a[2], a[3])
  ctx.strokeStyle = 'rgba(0, 0, 0, 1)' //max = 255
  ctx.stroke()
  ctx.closePath()

  ctx.beginPath()
  var a = to_board_rect(-BOARD_RANGE, BOARD_RANGE, BOARD_RANGE * 2, BOARD_RANGE * 2)
  ctx.rect(a[0], a[1], a[2], a[3])
  ctx.strokeStyle = 'rgba(220, 220, 220, 1)' //max = 255
  ctx.stroke()
  ctx.closePath()

  //draw targets
  trackers = trackers_arr[selected_target_index]
  function draw_target(name, color) {
    var target_size = 0.07
    ctx.beginPath()
    ctx.arc(to_board_x(trackers[name][0]), to_board_y(trackers[name][1]), to_board_x_length(target_size), 0, Math.PI * 2);
    ctx.fillStyle = color
    ctx.fill()
    ctx.closePath()
  }

  draw_target('red', '#EE3333')
  draw_target('yellow', '#D9D933')

  //draw robot
  var robot_size = 0.085
  var robot_angle = (bearing - 90) / 180 * Math.PI
  ctx.beginPath()
  ctx.arc(to_board_x(center_x), to_board_y(center_y), to_board_x_length(robot_size), 0, Math.PI * 2);
  ctx.fillStyle = '#0095DD'
  ctx.fill()
  ctx.closePath()

  const cam_angle = Math.PI / 8.0
  ctx.beginPath()
  ctx.moveTo(to_board_x(center_x), to_board_y(center_y))
  ctx.arc(to_board_x(center_x), to_board_y(center_y), to_board_x_length(robot_size), robot_angle - cam_angle, robot_angle + cam_angle)
  ctx.lineTo(to_board_x(center_x), to_board_y(center_y))
  ctx.fillStyle = '#004399'
  ctx.fill()
  ctx.closePath()

  if (overlay)
  {
    if (overlay_alpha_increasing)
    {
      overlay_alpha += 0.006
    }
    else 
    {
      overlay_alpha -= 0.006
    }
    if (overlay_alpha > 1.5) {
      overlay_alpha_increasing = false
      overlay_alpha = 1.5
    }
    if (overlay_alpha < -1) {
      overlay_alpha_increasing = true
      overlay_alpha = -1
    }

    ctx.globalAlpha = Math.min(1, Math.max(0, overlay_alpha))
    ctx.drawImage(overlay_image, 0, 0)
    ctx.globalAlpha = 1.0
    ctx.drawImage(overlay_text_image, 0, 0)
  }
}


function update_camera_image()
{
  if (update_camera_image_on_next_tick || last_ten_times.length < 10)
  {
    var start = Date.now()
    update_camera_image_on_next_tick = false
    var z_val = []
    const rebuilt_big_td = tf.tidy(() => {
      var zs
      if (load_slider_values_on_next_tick)
      {
        var zs = []
        for (var zi = 0; zi < 8; zi++) {
          var vl = sliders[zi].value
          vl = vl / 100.0
          vl *= (MAX_LATENT * 2.0)
          vl -= MAX_LATENT
          zs.push(vl)
        }
        zs = tf.tensor([zs])
      }
      else
      {
        zs = inference(tf.tensor([[center_x, center_y, Math.sin(bearing / 180.0 * Math.PI), Math.cos(bearing / 180.0 * Math.PI)]]), 
                                  var_list, (model_json.latent.length * selected_target_index))
      }
      z_val = zs.dataSync()
      const recs = reverse_parse(decoder(tf.tensor([z_val]), var_list, (model_json.latent.length * model_json.trackers.length)))
      const [r, g, b] = tf.unstack(recs, 3)
      const a = tf.cast(tf.mul(tf.ones(r.shape), tf.tensor(255)), 'int32')
      const rebuilt = tf.stack([r, g, b, a], 3) //BATCH, H, W, C
      const rebuilt_big = tf.image.resizeNearestNeighbor(rebuilt, [CAMERA_HEIGHT, CAMERA_WIDTH])
      return rebuilt_big
    })
    
    camera_image = new ImageData(new Uint8ClampedArray(rebuilt_big_td.dataSync()), CAMERA_WIDTH, CAMERA_HEIGHT)

    camera_ctx.putImageData(camera_image, 0, 0)

    if (!load_slider_values_on_next_tick)
    {
      for (var zi = 0; zi < z_val.length; zi++)
      {
        var vl = (z_val[zi] + MAX_LATENT) / (MAX_LATENT * 2.0)
        vl = vl * 100
        sliders[zi].value = vl
      }
    }
    update_camera_image_on_next_tick = false
    load_slider_values_on_next_tick = false
    end = Date.now() - start
    last_ten_times.push(end)
    if (last_ten_times.length > 10)
    {
      last_ten_times.shift()
    }
    if (last_ten_times.length == 10)
    {
      var sum = 0
      for (var i = 0; i < last_ten_times.length; i++)
      {
        sum += last_ten_times[i]
      }
      sum /= 10.0
      if (sum > 50)
      {//50ms is a bit slow
        speed_scale = (sum / 50.0)
        speed_scale = Math.min(speed_scale, 3)
        if (!warning_up && tf.getBackend() != 'webgl')
        {
          warning_up = true
          document.getElementById('warn').style.display = 'block'
        }
      }
      else
      {
        speed_scale = 1.0
      }
    }
  }
}


function gameloop() {
  
  const angle_speed = 4 * speed_scale
  const movement_speed = 0.015 * speed_scale
  const delta_x = movement_speed * Math.sin(bearing / 180.0 * Math.PI)
  const delta_y = movement_speed * Math.cos(bearing / 180.0 * Math.PI)
  if (upPressed)
  {
    center_x += delta_x
    center_y += delta_y
    update_camera_image_on_next_tick = true
    overlay = false
  }
  if (downPressed)
  {
    center_x -= delta_x
    center_y -= delta_y
    update_camera_image_on_next_tick = true
    overlay = false
  }
  if (rightPressed)
  {
    bearing += angle_speed
    update_camera_image_on_next_tick = true
    overlay = false
  }
  if (leftPressed)
  {
    bearing -= angle_speed
    update_camera_image_on_next_tick = true
    overlay = false
  }

  draw();
}

function slide_change()
{
  load_slider_values_on_next_tick = true
  update_camera_image_on_next_tick = true
}

function load1() {
  selected_target_index = 1 - 1
  update_camera_image_on_next_tick = true
}
function load2() {
  selected_target_index = 2 - 1
  update_camera_image_on_next_tick = true
}
function load3() {
  selected_target_index = 3 - 1
  update_camera_image_on_next_tick = true
}
function load4() {
  selected_target_index = 4 - 1
  update_camera_image_on_next_tick = true
}
function load5() {
  selected_target_index = 5 - 1
  update_camera_image_on_next_tick = true
}
function load6() {
  selected_target_index = 6 - 1
  update_camera_image_on_next_tick = true
}

var model_json
var var_list
var trackers_arr
async function run() {
  //tf.setBackend('cpu');
  const [model_json, var_list, trackers_arr] = await download_weights()
  window.trackers_arr = trackers_arr
  window.var_list = var_list
  window.model_json = model_json

  for (var zi = 0; zi < 8; zi++)
  {
    sliders.push(document.getElementById('lat' + (zi + 1)))
  }
  overlay_image = new Image();
  overlay_image.src = "arrow5.png"
  overlay_text_image = new Image();
  //overlay_text_image.src = "arrow4.png"

  canvas = document.getElementById('boardCanvas')
  ctx = canvas.getContext('2d')
  camera_canvas = document.getElementById('cameraCanvas')
  camera_ctx = camera_canvas.getContext('2d')
  setInterval(gameloop, 10)
  setInterval(update_camera_image, 10)
  
}

document.addEventListener('DOMContentLoaded', run)