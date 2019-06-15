<template>
  <div class="container">
    <div class="jumbotron">
      <h1 class="display-4">Image Captioning with Emotion</h1>
      <p class="lead">
        Generating image caption with emotion in bahasa. This project is intended for final project
        in
        Informatics Bandung Institute of Technology
      </p>
      <hr class="my-4">
      <p>Dery Rahman Ahaddienata - 13515097 © 2019</p>
      <a class="btn btn-secondary" href="#" role="button">Github</a>
    </div>
    <div class="row">
      <div class="col-8">
        <div class="form-group row">
          <label for="imagePath" class="col-sm-2 col-form-label">Image Path</label>
          <div class="col-sm-10">
            <div class="input-group">
              <div class="custom-file">
                <input
                  type="file"
                  class="custom-file-input"
                  id="inputFile"
                  @change="handleFileUpload($event)"
                >
                <label class="custom-file-label" for="inputFile">{{ filename }}</label>
              </div>
              <div class="input-group-append">
                <button class="btn btn-secondary" type="button" @click="generate()">Generate</button>
              </div>
            </div>
          </div>
        </div>

        <fieldset class="form-group">
          <div class="row">
            <legend class="col-form-label col-sm-2 pt-0">Mode</legend>
            <div class="col-sm-10">
              <div class="form-check form-check-inline">
                <input
                  class="form-check-input"
                  type="radio"
                  name="mode"
                  id="modeFactual"
                  value="factual"
                  v-model="mode"
                  checked
                >
                <label class="form-check-label" for="modeFactual">Factual</label>
              </div>
              <div class="form-check form-check-inline">
                <input
                  class="form-check-input"
                  type="radio"
                  name="mode"
                  id="modeHappy"
                  value="happy"
                  v-model="mode"
                >
                <label class="form-check-label" for="modeHappy">Happy</label>
              </div>
              <div class="form-check form-check-inline">
                <input
                  class="form-check-input"
                  type="radio"
                  name="mode"
                  id="modeSad"
                  value="sad"
                  v-model="mode"
                >
                <label class="form-check-label" for="modeSad">Sad</label>
              </div>
              <div class="form-check form-check-inline">
                <input
                  class="form-check-input"
                  type="radio"
                  name="mode"
                  id="modeAngry"
                  value="angry"
                  v-model="mode"
                >
                <label class="form-check-label" for="modeAngry">Angry</label>
              </div>
            </div>
          </div>
        </fieldset>

        <div class="form-group row">
          <label for="status" class="col-form-label col-sm-2 pt-0">Status</label>
          <div class="col-sm-10">
            <span class="badge badge-pill badge-primary" v-if="status == 'loading'">loading...</span>
            <span class="badge badge-pill badge-danger" v-else-if="status == 'error'">error</span>
            <span class="badge badge-pill badge-success" v-else-if="status == 'done'">done!</span>
            <span class="badge badge-pill badge-secondary" v-else>-</span>
          </div>
        </div>
      </div>
      <div class="col-4">
        <img class="rounded float-right img-thumbnail" :src="result.path_img" alt=" Card image cap">
      </div>
    </div>
    <div class="row my-5">
      <div class="col-12">
        <ul class="list-group">
          <li class="list-group-item d-flex justify-content-between align-items-center">
            {{ result.nic }}
            <span class="badge badge-primary">NIC</span>
          </li>
          <li class="list-group-item d-flex justify-content-between align-items-center">
            {{ result.nic_att }}
            <span class="badge badge-danger">NIC+Att</span>
          </li>
          <li class="list-group-item d-flex justify-content-between align-items-center">
            {{ result.stylenet }}
            <span class="badge badge-primary">StyleNet</span>
          </li>
          <li class="list-group-item d-flex justify-content-between align-items-center">
            {{ result.stylenet_att }}
            <span class="badge badge-danger">StyleNet+Att</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap-vue/dist/bootstrap-vue.css';

export default {
  name: 'Main',
  data() {
    return {
      file: '',
      mode: 'factual',
      status: 'init',
      filename: 'Choose file',
      result: {
        nic: '-',
        nic_att: '-',
        stylenet: '-',
        stylenet_att: '-',
        path_img:
          'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22286%22%20height%3D%22180%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20286%20180%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_16b56db6f13%20text%20%7B%20fill%3Argba(255%2C255%2C255%2C.75)%3Bfont-weight%3Anormal%3Bfont-family%3AHelvetica%2C%20monospace%3Bfont-size%3A14pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_16b56db6f13%22%3E%3Crect%20width%3D%22286%22%20height%3D%22180%22%20fill%3D%22%23777%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2298.6328125%22%20y%3D%2296.6046875%22%3EImage%20cap%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E',
      },
    };
  },
  methods: {
    handleFileUpload(event) {
      this.file = event.target.files[0];
      this.filename = this.file.name;
    },
    generate() {
      const formData = new FormData();
      formData.append('file', this.file);
      this.status = 'loading';
      console.log(process.env);
      axios
        .post(
          `${process.env.VUE_APP_BACKEND_HOST}/generate?mode=${this.mode}`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          },
        )
        .then((response) => {
          console.log(response.data);
          this.status = 'done';
          this.result = response.data;
          this.result.path_img = process.env.VUE_APP_BACKEND_HOST + response.data.path_img;
        })
        .catch((error) => {
          this.status = 'error';
          if (error.response) {
            alert(error.response.data);
          } else {
            alert(error.message);
          }
        });
    },
  },
};
</script>
