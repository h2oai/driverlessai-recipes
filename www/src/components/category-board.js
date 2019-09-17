import React from "react"
import PropTypes from "prop-types"
import Switch from '@material-ui/core/Switch';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import { Container } from "@material-ui/core";
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
  topControl : {
    textAlign: 'center',
    marginBottom: '40px',
  },
}))

const CategoryBoard = ({ categories, toggleFunc }) => {

  const onCategoryClick = (name) => (e) => { 
    toggleFunc(name)
  }
  
  const classes = useStyles();

  return (
    <Container maxWidth="sm" className={classes.topControl}>
      <FormControl component="fieldset">
        <FormGroup aria-label="position" row>
          {Object.keys(categories).map((category, index) => (
          <FormControlLabel
          key={category}
          value={category}
          control={<Switch key={category} color="primary" onChange={onCategoryClick(category)} checked={categories[category]} />}
          label={category}
          labelPlacement="top"
          />
          ))}
        </FormGroup>
      </FormControl>
    </Container>
  ) 
}

CategoryBoard.propTypes = {
  categories: PropTypes.object.isRequired,
  toggleFunc: PropTypes.func.isRequired,
}

export default CategoryBoard
