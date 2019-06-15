var app = new Vue({
    el: '#app',
    data: {
        file: '',
        mode: 'factual',
        status: 'init',
        result: {
            nic: '-',
            nic_att: '-',
            stylenet: '-',
            stylenet_att: '-',
            path_img: 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22286%22%20height%3D%22180%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20286%20180%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_16b56db6f13%20text%20%7B%20fill%3Argba(255%2C255%2C255%2C.75)%3Bfont-weight%3Anormal%3Bfont-family%3AHelvetica%2C%20monospace%3Bfont-size%3A14pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_16b56db6f13%22%3E%3Crect%20width%3D%22286%22%20height%3D%22180%22%20fill%3D%22%23777%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2298.6328125%22%20y%3D%2296.6046875%22%3EImage%20cap%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E'
        }
    },
    methods: {
        handleFileUpload(event) {
            this.file = event.target.files[0]
        },
        generate() {
            formData = new FormData()
            formData.append('file', this.file)
            this.status = 'loading'

            axios.post('http://localhost:5000/generate?mode=' + this.mode,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                }
            ).then(response => {
                console.log(response.data)
                this.status = 'done'
                this.result = response.data
            }).catch(error => {
                this.status = 'error'
                if (error.response) {
                    alert(error.response.data)
                } else {
                    alert(error.message)
                }
            })
        }
    }
})


$('#inputFile').on('change', function () {
    //get the file name
    var fileName = $(this).val();
    fileName = fileName.replace('C:\\fakepath\\', " ")
    //replace the "Choose a file" label
    $(this).next('.custom-file-label').html(fileName);
})